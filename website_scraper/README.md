FlemishWebsiteScraper is a simple scraper to scrape the HTML code of the homepage of websites.
The script will read a .csv file containing the URLs to be scraped.
The dataset will be processed in batches of a batch size to be specified by the user. The script will then scrape the websites using multiple threads (number of threads can be specified by user).
After scraping a batch the results will be stored in a .xlsx file which will be uploaded to Google Cloud.

The settings.ini file contains all adjustable parameters and must be stored together with the main.py file. The location of the dataset, the credentials for the Google Cloud upload and the desired output location can be specified in the .ini file
as well as the number of threads to be used, the batch size and the last completed batch number in case the scripts needs to be restarted after an interruption. 
