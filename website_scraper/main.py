import pandas as pd

import requests
from tqdm import tqdm
import os

from bs4 import BeautifulSoup
import langdetect
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import itertools
import configparser
import time
'''import fastparquet
import zipfile
import file_manipulator'''

#import tkinter as tk
#from tkinter import filedialog, messagebox

from google.cloud import storage

# Function to extract domain from URL
def get_domain(url):
  parsed_url = urlparse(url)
  return f"{parsed_url.scheme}://{parsed_url.netloc}/"


# Function to scrape website content
def scrape_website(url, session):
  try:
    response = session.get(str(url), timeout=(5, 10))
    if response.status_code == 200 or response.status_code == 304:
      text = BeautifulSoup(response.text, 'html.parser')
      return text
    else:
      print(f"Request to {url} failed with status code {response.status_code}")
      return None
  except Exception as e:
    print(f"Error occurred while scraping {url}: {e}")
    return None


# Function to process each URL
# scrape for dutch text (nl) and afrikaans (af) as langdetext often misclassifies dutch texts as afrikaans.
def process_url(row, session, language=['nl', 'af']):
  url = get_domain(row.URL)
  try:
    text = scrape_website(url, session)
    if text is not None:
      lang = langdetect.detect(str(text.get_text(separator=' ', strip=True)))
      if lang in language:
        return row.KboNr, text
      else:
        dutch_url = find_dutch_variant(text)
        if dutch_url:
          print(f"Dutch variant of {url} found: {dutch_url}")
          dutch_text = scrape_website(dutch_url, session)
          return row.KboNr, dutch_text
        else:
          print(f"This website {url} is in {lang}")
          return row.KboNr, None
  except langdetect.lang_detect_exception.LangDetectException as e:
    print(f"Error occured while processing {url}: {e}")
    return row.KboNr, None


# Function to find Dutch variant of a website
def find_dutch_variant(soup):
  link_tag = soup.find('link', {'rel': 'alternate', 'hreflang': 'nl'})
  if link_tag:
    dutch_url = link_tag.get('href')
    return dutch_url
  else:
    return None

# Main function
def main():

  settings = read_settings()
  print(settings)

  csv_data_path = settings['csv_dataset_location']
  json_credentials_path = settings['json_credentials_path']
  local_storage_location_path = settings['local_storage_path']


  # Name of google cloud bucket
  bucket_name = "website_text_data"

  # Index is 0, always keep at 0
  index = 0

  # In case of a problem where the scraping got interrupted, change this parameter to the last fully scraped batch number.
  # If the error occurred during batch 5 than the last scraped batch is 4 which is what should be entered here.
  # This will ensure that if scraping is interrupted it can be restarted without having to overdo it all.
  # The default value is -1!
  # !!! WARNING, this only works if the batch_processing_size remains UNCHANGED!!
  last_fully_scraped_batch = int(settings['last_completed_batch'])

  # +/- 300.000 urls in dataset.
  # Splitten, recommend using a value  between 5000 (+60 files) and 50.000 (+6 files)
  # This processins size will dictate how many lines in each dataframe.
  batch_processing_size = int(settings['batch_size'])

  # Setting for saving as parquet file or .xlsx. (Not a Boolean because reading boolean using configparser required to much rewriting of the read_settings function for just one simple setting.)
  store_parquet_string = str(settings['parquet'])

  # Time out parameters for Google Cloud connection and upload.
  connect_time_out = int(settings['connect_time_out'])
  upload_time_out = int(settings['upload_time_out'])

  store_parquet = False
  if store_parquet_string == "True":
    store_parquet = True

  #Credentials for uploading to Google Cloud
  if not os.path.exists(json_credentials_path):
    raise FileNotFoundError("The JSON credentials file could not be founnd. Make sure that the given path in the settings.ini file is correct!")

  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_credentials_path

  # The following lines of code are used to test if credentials are correct and uploading works.
  # If not the code is stopped from executing
  current_path = os.getcwd()
  settings_path = os.path.join(current_path, 'settings.ini')

  upload_succes = test_cloud_upload(settings_path, bucket_name)
  if not upload_succes:
    raise ConnectionError("Failed to upload to Google Cloud. This error needs to be fixed before continuing!")
  else:
    print("Uploading test to Google Cloud success")

  # Check if checkpoints and output folder exist. If not make folder.
  output_path_folder = os.path.join(local_storage_location_path, "output_dataset")

  if not os.path.exists(output_path_folder):
    os.mkdir(output_path_folder)

  log_path =os.path.join(output_path_folder, 'logs.txt')
  log_dict = dict()

  if not os.path.exists(csv_data_path):
    raise FileNotFoundError("The dataset file containing all URLS could not be found. Make sure the path in the .ini file is correct")

  urldata_complete = list(pd.read_csv(csv_data_path,
                                   chunksize=batch_processing_size))  # [['KboNr','URL']]

  # count the total number of batches for progress tracking
  total_number_batches = len(urldata_complete)-1

  for urldata in urldata_complete:
    batch_dict = dict()
    log_dict[f'batch_number_{index}'] = batch_dict

    if index - 1 < last_fully_scraped_batch:
      print(f"skipping batch {index} out of {total_number_batches}. ")
      index += 1
      continue

    print(f'Processing batch {index} out of {total_number_batches}. ({index}/{total_number_batches})')
    # create correct path to store al files with scraped data
    output_path = os.path.join(output_path_folder, f"textdata_{index}.xlsx")

    textdata = pd.DataFrame({'KboNr': urldata['KboNr'], 'Text': [None] * len(urldata), 'URL': urldata['URL']})
    # textdata = pd.DataFrame({'KboNr': urldata['KboNr'], 'Text': [None] * len(urldata)})

    num_threads = int(settings['threads']) #--> change to 1 for sequential scraping

    with requests.Session() as session:
      with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = []
        # Process URLs using ThreadPoolExecutor
        for result in tqdm(executor.map(process_url, textdata.itertuples(), itertools.repeat(session)),
                           total=len(textdata), desc="Processing sites", unit='site', leave=False):
          results.append(result)

    # Update textdata with scraped text
    for result in results:
      if result is not None:
        kbonr, text = result
        if text is not None:
          textdata.loc[textdata['KboNr'] == kbonr, 'Text'] = str(text)

    scraped_percentage = textdata['Text'].count() / len(textdata) * 100
    print("PERCENTAGE SCRAPED: " + str(scraped_percentage) + "%")

    batch_dict['scraped_percentage'] = str(scraped_percentage)

    # Save results to Excel file
    #remove urls from dataframe before storing in xlsx
    textdata_no_url = textdata.drop('URL', axis=1)
    #textdata.to_excel(output_path, index=False) #use only for testing

    #Store in parquet or xlsx file and push go google cloud using upload_to_gcs function
    try:
      if store_parquet:
        textdata_no_url.to_parquet(os.path.join(output_path_folder, f"textdata_{index}.parquet")) #,force_ascii=False) #compression='gzip')#, index=False)
      else:
        textdata_no_url.to_excel(output_path, index=False)
      batch_dict['local_storage_success'] = 'True'
      batch_dict['local_storage_error'] = 'None'
    except Exception as e:
      batch_dict['local_storage_success'] = 'False'
      batch_dict['local_storage_error'] = e

    try:
      if store_parquet:
        upload_time = upload_to_gcs(os.path.join(output_path_folder, f"textdata_{index}.parquet"), bucket_name,
                      f'textdata_{index}.parquet', connect_time_out, upload_time_out)
      else:
        print('WARNING! Saving as .xlsx file. This is not recommended.')
        upload_time = upload_to_gcs(output_path, bucket_name,
                      f'textdata_{index}.xlsx', connect_time_out, upload_time_out)
      batch_dict['cloud_storage_success'] = 'True'
      batch_dict['cloud_storage_error'] = 'None'
      batch_dict['upload_time'] = upload_time
    except Exception as e:
      batch_dict['cloud_storage_success'] = 'False'
      batch_dict['cloud_storage_error'] = str(e)
      batch_dict['upload_time'] = -1
      print(f'Uploading of data_file{index} failed becuase of {e}')
    write_logs(log_dict, log_path, bucket_name)
    index += 1

  write_logs(log_dict, log_path, bucket_name)

# Function to upload file to Google Cloud Storage
def upload_to_gcs(local_file_path, bucket_name, destination_blob_name, connect_time_out = 10, upload_time_out = 60):
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)
  start_time = time.time()
  try:
    blob.upload_from_filename(local_file_path, timeout=(connect_time_out,upload_time_out)) #!! Timeout for upload
  except TimeoutError:
    print("Time out error")
  except Exception as e:
    raise e
  end_time = time.time()
  processing_time = end_time - start_time
  print(f"File uploaded to {destination_blob_name} in bucket {bucket_name} in {processing_time} seconds.")
  return processing_time

# Read settings file
def read_settings():
  config = configparser.ConfigParser()
  config.read("settings.ini")
  settings = {}
  for section in config.sections():
    for key, value in config.items(section):
      settings[key] = value
  return settings

def write_logs(log_dict, log, bucket_name):
  with open(log, 'w') as f:
    for key1, value1 in log_dict.items():
      f.write(('%s\n' % key1))
      for key2, value2 in value1.items():
        f.write(('\t%s:%s\n' % (key2, value2)))
  try:
    upload_to_gcs(log, bucket_name, 'log_file.txt')
  except Exception as e:
    print(e)


# Function to do a test upload to Google Cloud (to be called before starting the scraping)
def test_cloud_upload(settings_path, bucket_name):
  try:
    upload_to_gcs(settings_path, bucket_name, f'TEST_CLOUD_UPLOAD')
    return True
  except:
    return False

if __name__ == "__main__":
  main()
