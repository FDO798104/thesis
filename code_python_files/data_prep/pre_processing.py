from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
import hashlib
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import subprocess

subprocess.run(["pip", "install", "spacy"])
subprocess.run(["python", "-m", "spacy", "download", "nl_core_news_sm"])
import spacy
nltk.download('punkt')
nltk.download('stopwords')

def main():
  # Does the loaded textdata contain scraped HTML code? --> set to True.
  extract_text_from_html = False
  tags_to_extract = ['p','h','li','span','td']
  backup_files = True

  # Set to True if you want to handle duplicates in any way. If set to False than
  # two settings below will be ignored.
  handle_duplicates = False
  # Set use_duplicate_removal_heuristic to True for heuristic
  use_duplicate_removal_heuristic = False
  # Set to True if you want to delete all duplicates instead of using heuristic
  remove_all_duplicates = True

  # Set to True if you want to remove rows with to few words. If use_duplicate_removal_heurstic is used than all rows with less than 10 words will already be removed.
  do_remove_rows_with_to_few_words = False

  # Set to number of words you want a row to have minimum
  min_words = 20

  # Downcase the text
  do_downcasing = False

  # Remove all numbers from text
  do_remove_numbers = False

  # Remove puntuation from text
  do_remove_punctuation = False

  # Remove stopwords
  do_remove_stopwords = False

  # Remove set of words related to cookies message or javascript not activated banner
  do_remove_cookies_etc_message = False
  do_remove_java = False

  # Remove single characters and extra white spaces
  do_remove_noise = True

  # Remove the rows where no NACE code is linked with the DATA
  # !! Can't think of a single reason to set this to False !!
  do_remove_null_nace = False

  # Create ngrams after pre processing
  create_ngrams = True

  # Set range for ngrams
  ngram_range = (3,6)

  # Use Snowball Stemmer
  use_snowball_stemmer = False

  # Use lemmatization
  lemmatize_text = False

  run_pre_processing(extract_text_from_html,
                    tags_to_extract,
                    handle_duplicates,
                    use_duplicate_removal_heuristic,
                    remove_all_duplicates,
                    do_remove_rows_with_to_few_words,
                    min_words,
                    do_downcasing,
                    do_remove_numbers,
                    do_remove_punctuation,
                    do_remove_stopwords,
                    do_remove_cookies_etc_message,
                    do_remove_java,
                    do_remove_noise,
                    do_remove_null_nace,
                    create_ngrams,
                    ngram_range,
                    use_snowball_stemmer,
                    lemmatize_text,
                    backup_files)

def run_pre_processing(do_extract_text_from_html,
                       tags_to_extract,
                       handle_duplicates,
                       use_duplicate_removal_heuristic,
                       remove_all_duplicates,
                       do_remove_rows_with_to_few_words,
                       min_words,
                       do_downcasing,
                       do_remove_numbers,
                       do_remove_punctuation,
                       do_remove_stopwords,
                       do_remove_cookies_etc_message,
                       do_remove_java,
                       do_remove_noise,
                       do_remove_null_nace,
                       create_ngrams,
                       ngram_range,
                       use_snowball_stemmer,
                       lemmatize_text,
                       backup_files):
  warning = False

  if not os.path.isdir(files_directory):
    raise FileNotFoundError(f"The value in original_files_directory is not an existing directory \n {files_directory}")

  if not os.path.isdir(files_directory):
    os.mkdir(files_directory)

  elif not os.path.isdir(backup_files_directory):
    os.mkdir(backup_files_directory)


  conflict_1 = sum([use_duplicate_removal_heuristic, remove_all_duplicates])
  conflict_2 = sum([create_ngrams, use_snowball_stemmer, lemmatize_text])

  if handle_duplicates and conflict_1 > 1:
    raise ValueError("There is a conflict in the settings. \'use_duplicate_removal_heuristic\' is set to \'True\' and \'remove_all_duplicates\' is set to \'True\'. \n \
    If you want to use duplicate removal heuristic set \'delete_all_duplicates\' to \'False\'\n")

  if handle_duplicates and conflict_1 == 0:
    raise ValueError("There is a conflict in the settings. \'use_duplicate_removal_heuristic\' is set to \'False\' and \'remove_all_duplicates\' is set to \'False\'. \n \
    \'handle_duplicates\' is set to \'True\'. One handling method must be chosen or \'handle_duplicates\' must be set to \'False\'.\n")
    warning = True

  if conflict_2 > 1:
    raise ValueError(f"There is a conflict in the settings. Of the following settings only 1 can be set to \'True\': \n \
    \'create_ngrams\' is set to \'{create_ngrams}\' \n \
    \'snowball_stemmer\' is set to \'{use_snowball_stemmer}\' \n \
    \'lemmatize_text\' is set to \'{lemmatize_text}\' \n \
    Correct this mistake before proceeding!")

  if not do_remove_null_nace:
    print(f'WARNING - The \'do_remove_null_nace\' setting is set to \'{do_remove_null_nace}\'.\n\
    Make sure that this is correct as not removing NULL values in the NACE columns will cause errors when training a model. \n')
    warning = True

  if do_remove_java and not do_remove_cookies_etc_message:
    print(f"If /'do_remove_cookies_etc_message/' is set to /'False/' the /'do_remove_jave/' setting will be ignored")
    warning = True

  if warning:
    print('\n \n')
    input_answer = input(f"WARNING: A deviation from the recommended settings was detected. Read the warnings above. Do you wish to ignore these warnings and continue? (yes/no)")
    if input_answer.lower() != 'yes':
      raise RuntimeError("Execution halted due to warnings.")
    else:
      print("Continuing despite warnings")

  if backup_files:
    for file_name, file_path in get_files().items():
        data = read_file(file_path)
        store_file(data, file_name, backup=True)

  input_answer = input("Directories reset to values provided in settings. If you have already done part of the pre processing make sure you select the directory to the partly pre processed files! Do you wish to continue? (yes/no)")
  if input_answer.lower() != 'yes':
    raise RuntimeError("Execution halted due to warnings.")
  elif warning:
    print("Continuing despite warnings")

  if do_extract_text_from_html:
    print("Extracting texts...")
    extract_text_from_html(tags_to_extract)
    print("Extracting texts complete")

  if handle_duplicates:
    print(f'Handling duplicates')
    duplicates_handler(use_duplicate_removal_heuristic, remove_all_duplicates)

  if do_downcasing:
    print("Reading files to downcase text...")
    for file_name, file_path in get_files().items():
      data = read_file(file_path)
      data['Text'] = data['Text'].apply(downcase_text)

      store_file(data, file_name, backup=False)
      del data
    print("\nDOWNCASING text completed\n")

  if do_remove_numbers:
    print("Reading files to remove numbers...")
    for file_name, file_path in get_files().items():
      data = read_file(file_path)
      data['Text'] = data['Text'].apply(remove_numbers)
      store_file(data, file_name, backup=False)
      del data
    print("\REMOVING numbers in text completed\n")

  if do_remove_punctuation:
    print("Reading files to remove punctuation...")
    for file_name, file_path in get_files().items():
      data = read_file(file_path)
      data['Text'] = data['Text'].apply(remove_punctuations)
      store_file(data, file_name, backup=False)
      del data
    print("\REMOVING punctuation in text completed\n")

  if do_remove_cookies_etc_message:
    print("Reading files to remove copyright and cookies message...")
    for file_name, file_path in get_files().items():
      data = read_file(file_path)
      data['Text'] = data['Text'].apply(remove_cookies_copyright)
      if do_remove_java:
        print("Removing JAVA message")
        data['Text'] = data['Text'].apply(remove_javascript)
      store_file(data, file_name, backup=False)
      del data
    print("\Removing copyright and cookies in text completed \n")

  if do_remove_noise:
    print("Reading files to remove noise...")
    for file_name, file_path in get_files().items():
      data = read_file(file_path)
      data['Text'] = data['Text'].apply(remove_noise)
      store_file(data, file_name, backup=False)
      del data
    print("\REMOVING noise in text completed\n")

  if do_remove_null_nace:
    print("Reading files to remove null nace rows...")
    for file_name, file_path in get_files().items():
      data = read_file(file_path)
      data = remove_null_nace_rows(data)
      store_file(data, file_name, backup=False)
      del data
    print("\REMOVING null nace rows in text completed\n")


  if do_remove_rows_with_to_few_words:
    count = 0
    print("Reading files to remove rows with to few words...")
    for file_name, file_path in get_files().items():
      data = read_file(file_path)
      data, count = count_words(data, count, min_words)
      store_file(data, file_name, backup=False)
      del data
    print("\REMOVING rows with to few words completed\n")

  #Remove stopwords from text #DONE
  def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords_dutch]
    return ' '.join(words)


  if do_remove_stopwords:
    nltk.download('stopwords')
    nltk.download('punkt')
    stopwords_dutch = set(stopwords.words('dutch'))
    print("Reading files to remove stopwords in text...")
    for file_name, file_path in get_files().items():
      data = read_file(file_path)
      data['Text'] = data['Text'].apply(remove_stopwords)
      store_file(data, file_name, backup=False)
      del data
    print("\REMOVING stopwords from text completed\n")

  if create_ngrams:
    print("Reading files to create ngrams...")
    for file_name, file_path in get_files().items():
      print(f'processing {file_name}')
      data = read_file(file_path)
      data['Text'] = data['Text'].apply(tokenize_ngrams)
      store_file(data, file_name, backup=False)
      del data
    print("\Creating ngrams text completed\n")

  if lemmatize_text:
    nlp = spacy.load("nl_core_news_sm")
    print("Reading files to lemmatize...")
    for file_name, file_path in get_files().items():
      print(f'processing {file_name}')
      data = read_file(file_path)
      data = lemmatize_dutch_text(data, nlp, 'Text')
      store_file(data, file_name, backup=False)
      del data
    print("\Lemmatizing text completed\n")

  if use_snowball_stemmer:
    stemmer = SnowballStemmer("dutch")
    print("Reading files to stem text...")
    for file_name, file_path in get_files().items():
      print(f'processing {file_name}')
      data = read_file(file_path)
      data = stem_dutch_text(data, stemmer, 'Text')
      store_file(data, file_name, backup=False)
      del data
    print("\Stemming text completed\n")

print("Process finished")

def get_files(overwrite = False, new_path = None):
  files_dict = dict()
  files_directory_temp = files_directory
  if overwrite and new_path != None:
    files_directory_temp = new_path

  for file_name in os.listdir(files_directory_temp):
    if file_name.endswith('.parquet'):
      file_path = os.path.join(files_directory_temp, file_name)
      files_dict[file_name] = file_path
  return files_dict

# This function reads a file path and returns the dataframe
def read_file(file_path, all=True):
  print("Processing file:", file_path)
  if file_path.endswith('.parquet'):
    data = pd.read_parquet(file_path)
  return data

def store_file(data, file_name, backup=False):
  if backup:
    directory_temp = backup_files_directory
  else:
    directory_temp = files_directory
  save_path = os.path.join(directory_temp, file_name)
  data.to_parquet(save_path)
  print("File processed:", save_path)

# Extract text with certain tags #DONE
def extract_text(html, tags_to_extract):
  try:
    del extracted_text
  except:
    None
  soup = BeautifulSoup(html, 'html.parser')
  extracted_text = []
  for tag in tags_to_extract:
    extracted_text.extend([tag.get_text() for tag in soup.find_all(tag)])
  return ' '.join(extracted_text)


def extract_text_from_html(tags_to_extract):
  print("Reading files to extract text from HTML...")
  for file_name, file_path in get_files().items():
    data = read_file(file_path)
    data['Text'] = data['Text'].apply(lambda x: extract_text(x, tags_to_extract))
    store_file(data, file_name, backup=False)
    del data
  print("\nEXTRACTING text completed\n")

def hash_and_collect(data, total):
  data['Text_Hash'] = data['Text'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
  del data['Text']
  total = pd.concat([total,data], ignore_index = True)
  return total

def remove_selected_kbo(kbo_removal_list):
  remove_count = 0
  for file_name, file_path in get_files().items():
    data = read_file(file_path)
    before = data.shape[0]
    data = data[~data['KboNr'].isin(kbo_removal_list)]
    store_file(data, file_name, backup=False)
    after = data.shape[0]
    remove_count += (before-after)
    del data
  print("\nHANDLING DUPLICATES completed\n")



# duplicate_handler() should be called if duplicates need to be removed
# First all data will be loaded, hashed and added to the same dataframe
def duplicates_handler(use_duplicate_removal_heuristic, remove_all_duplicates):
  # The kbo removal list will contain all the KBOs we want to delete
  kbo_removal_list = []
  hashed_total_data = pd.DataFrame()
  # Load data, hash it and add to single dataframe
  for file_name, file_path in get_files().items():
    data = read_file(file_path)
    hashed_total_data = hash_and_collect(data, hashed_total_data)
    del data
  # Count how often a hash appears
  duplicate_counts = hashed_total_data['Text_Hash'].value_counts()
  # Create hashed_dict containing the hashes as key and the associated KBO's in a list as value
  hashes_dict = dict()
  for index, row in hashed_total_data.iterrows():
    current_hash = row['Text_Hash']
    current_KBO = row['KboNr']
    # Check if hash exists as key. If so add to existing list.
    if current_hash in hashes_dict.keys():
      hashes_dict[current_hash].append(current_KBO)
    else:
      hashes_dict[current_hash] = [current_KBO]
  if use_duplicate_removal_heuristic:
    kbo_removal_list = heuristic_removal(hashes_dict, kbo_removal_list)

  elif remove_all_duplicates:
    kbo_removal_list = remove_duplicates_complete(hashes_dict, kbo_removal_list)

  else:
    kbo_removal_list = remove_always(hashes_dict, kbo_removal_list)

  remove_selected_kbo(kbo_removal_list)

def remove_duplicates_complete(hashes_dict, kbo_removal_list):
  for hash, kbo_list in hashes_dict.items():
    if len(kbo_list) >=2:
      kbo_removal_list.extend(kbo_list)
  return kbo_removal_list

def remove_always(hashes_dict, kbo_removal_list):
  delete_all_instances = pd.read_excel('/content/drive/MyDrive/Thesis/duplicates.xlsx')
  # Create a list were we will store the bad KBO that acts as example.
  # Every hash in the hashed_total_data has a list of associated KBO
  # If a kbo found in this list is associated with a specific HASH,
  # than all KBO's of that hash need to be deleted
  example_bad_kbo = []
  for KBO in delete_all_instances['Example KBO']:
    example_bad_kbo.append(KBO)
  print(example_bad_kbo)
  for hash, kbo_list in hashes_dict.items():
  # Now find all hashes of the KBO's above and add all associated KBO's to kbo_removal_list
    for bad_kbo in example_bad_kbo:
      # Check if hash is 'bad'
      if bad_kbo in kbo_list:
        kbo_removal_list.extend(kbo_list)
    return kbo_removal_list

def heuristic_removal(hashes_dict, kbo_removal_list):
  # our delete heuristic:
  # Every hash appearing more than 10 times has been manually checked.
  # A number of hashes were completly wrong websites and thus need to be removed
  # completly
  delete_all_instances = pd.read_excel('/content/drive/MyDrive/Thesis/duplicates.xlsx')
  # Create a list were we will store the bad KBO that acts as example.
  # Every hash in the hashed_total_data has a list of associated KBO
  # If a kbo found in this list is associated with a specific HASH,
  # than all KBO's of that hash need to be deleted
  example_bad_kbo = []
  for KBO in delete_all_instances['Example KBO']:
    example_bad_kbo.append(KBO)
  print(example_bad_kbo)

  # Now we use the dataset containing a sample of some kbo's related to cities and other governments
  knu_students = pd.read_csv('/content/drive/MyDrive/Thesis/knu_students.csv')

  #Create list of all government KBO's we have available
  gov_kbo_list = list()
  for index, row in knu_students.iterrows():
    gov_kbo_list.append(row['KboNr'])


  for hash, kbo_list in hashes_dict.items():
    # Now find all hashes of the KBO's above and add all associated KBO's to kbo_removal_list
    bad_kbo_found = False
    for bad_kbo in example_bad_kbo:
      # Check if hash is 'bad'
      if bad_kbo in kbo_list:
        kbo_removal_list.extend(kbo_list)
        bad_kbo_found = True
    if not bad_kbo_found:
      kbo_gove_boolean = False
      # Check for all kbo in kbo_list of current hash if it is a government kbo. If it is than we want to keep only this KBO
      for kbo_to_be_handled in kbo_list:
        if kbo_to_be_handled in gov_kbo_list:
          kbo_list_copy = kbo_list.copy()
          kbo_list_copy.remove(kbo_to_be_handled)
          kbo_removal_list.extend(kbo_list_copy)
          kbo_gove_boolean = True # boolean to check if hash was already handled by this part
          break
      if not kbo_gove_boolean:
        if len(kbo_list) >= 10:
          amount_counter = 0
          for kbo_to_be_handled in kbo_list:
            if amount_counter >= 2:
              kbo_removal_list.append(kbo_to_be_handled)
            amount_counter += 1

        # Check if hash has has more than 1 kbo associated, if so handle it.
        elif 10 > len(kbo_list) >= 2:
          amount_counter = 0
          kbo_gove_boolean = False
        # if hash was not resolved by gov check, keep only the KBO number with lowest numerical value (We found that during random checks this was often the correct KBO)
          lowest_kbo = 99998765432101
          for kbo_to_be_handled in kbo_list:
            if int(kbo_to_be_handled) < lowest_kbo:
              lowest_kbo = int(kbo_to_be_handled)
          for kbo_to_be_handled in kbo_list:
            if int(kbo_to_be_handled) == lowest_kbo:
              continue
            else:
              kbo_removal_list.append(kbo_to_be_handled)

  kbo_removal_list = list(set(kbo_removal_list))
  print(f'Length after removing all hashes selected for total removal {len(kbo_removal_list)}')
  return kbo_removal_list

def downcase_text(text):
  return text.lower()

# Remove numbers from text
def remove_numbers(text):
  return re.sub(r'\d+','',text)

# Remove special characters and punctuations from text #DONE
def remove_punctuations(text):
  additional_characters = "'\"©...°”“‘’$€¬±¹£«®"
  all_characters = string.punctuation + additional_characters
  return ''.join(char for char in text if char not in all_characters)

# Function to remove JavaScript related words from text
def remove_javascript(text):
  javascript_words = ['javascript', 'js', 'script', 'document', 'function', 'var', 'let', 'const', 'window', 'alert',
                    'console', 'return', 'true', 'false', 'if', 'else', 'for', 'while', 'break', 'continue', 'try',
                    'catch', 'finally', 'throw', 'new', 'this', 'class', 'instanceof', 'typeof', 'delete', 'in',
                    'async', 'await', 'import', 'export', 'module', 'default', 'static', 'extends', 'super', 'break',
                    'case', 'switch', 'default', 'yield', 'debugger', 'Infinity', 'NaN', 'isFinite', 'isNaN',
                    'parseInt', 'parseFloat', 'undefined']

  for word in javascript_words:
    word_2 = " " + word + " "
    text = text.replace(word_2, ' ')
  return text

#Remove cookie/copyright related from text #DONE
def remove_cookies_copyright(text):
  cookies = ['accepteer', 'accepteren', 'accept', 'advertenties', 'advertisements', 'analyse', 'analyze', 'analytics',
            'analytische','analytisch','bepaalde', 'certain', 'bezoeker','visitor','belgian', 'browser','cookies','cookie','copyright',
            'choose','kies','kiezen', 'delen', 'share','derden','third', 'parties','party', 'disclaimer', 'functioneel','functional',
            'functionele','functioneren','function','gebruik', 'gebruiker','gebruikt', 'use', 'user','used','inhoud',
            'content','instellingen','settings', 'klikken','click','login','register','registreer','necessary','noodzakelijk',
            'noodzakelijke','opgeslagen', 'opslaan', 'save','saved','pagina','page','policy','beleid','privacy', 'relevante',
            'relevant','social','sociaal','sociale', 'store','opslaan','toestemming','consent', 'voorkeuren','preference',
            'preferences','website', 'websites','algemene','voorwaarden', 'aanmelden','account','gegevens', 'www', 'com',
            'contact', 'contacteer','websites','gebruiken','this', 'these', 'that', 'cookieverklaring', 'toggle', 'more', 'about', 'rights',
            'reserved', 'privacy', 'support','ondersteuning', 'copyright', 'cookiebeleid', 'nl', 'fr','eng','de', 'gdpr', 'sitemap']
  copyright = ['auteursrecht', 'auteursrechten', 'auteurswet', 'auteursrechtelijk', 'auteursrechtbescherming', 'auteursrechtinbreuk', 'copyrightmelding',
            'auteursrechtclaim', 'auteursrechtverklaring', 'auteursrechtvermelding','copyright']
  for word in cookies:
    word_2 = " " + word + " "
    text = text.replace(word_2,' ')
  for word in copyright:
    word_2 = " " + word + " "
    text = text.replace(word_2,' ')
  return text

#Remove single letters, ' and " from text #DONE
def remove_noise(text):
    # Remove single letters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove extra whitespaces and ensure only one whitespace between words
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_null_nace_rows(dataframe):
    # Create a mask where True indicates 'NACE' is not null
    mask = dataframe['NACE'].notnull()

    # Apply the mask to filter the dataframe
    filtered_dataframe = dataframe[mask]

    return filtered_dataframe

def count_words(dataframe, count,min_words):
    # Splitting text into words and counting the number of words in each row
    dataframe['word_count'] = dataframe['Text'].apply(lambda x: len(str(x).split()))
    shape_b = dataframe.shape[0]
    dataframe = dataframe[dataframe['word_count'] >= min_words]
    shape_a = dataframe.shape[0]
    count += (shape_b - shape_a)
    # Dropping the 'word_count' column as it's no longer needed
    dataframe = dataframe.drop(columns=['word_count'])

    return dataframe, count

def tokenize_ngrams(text):
  ngram_range = (3,6)
  if len(text) < max(ngram_range):
        return ''  # Return empty string if text length is insufficient for n-grams
  else:
      vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)
      try:
          X = vectorizer.fit_transform([text])
          if X.shape[1] == 0:
              return ''  # Return empty string if vocabulary is empty
          else:
              ngrams = vectorizer.get_feature_names_out()
              ngrams = [gram.replace(' ','_') for gram in ngrams]
              return ' '.join(ngrams)
      except ValueError:
          return ''  # Return empty string in case of ValueError

def lemmatize_dutch_text(dataframe, nlp, text_column='text'):

    # Define a lemmatization function that takes a document and returns its lemmatized version
    def lemmatize(doc):
        return " ".join([token.lemma_ for token in nlp(doc)])

    # Apply the lemmatization function to the specified text column
    dataframe[text_column] = dataframe[text_column].apply(lemmatize)

    return dataframe

def stem_dutch_text(dataframe, stemmer, text_column='Text'):
    # Define a stemming function that takes a document and returns its stemmed version
    def stem(text):
        words = nltk.word_tokenize(text)
        return " ".join([stemmer.stem(word) for word in words])

    # Apply the stemming function to the specified text column
    dataframe[text_column] = dataframe[text_column].apply(stem)

    return dataframe

if __name__ == "__main__":
    # This should be the path to the directory containing the data you want to preform pre processing on
  files_directory = '/content/drive/.../'

  # Set backup in main to true to first create copy of files before processing it. Copy will be stored in directory below.
  backup_files_directory = '/content/drive/.../'

  main()