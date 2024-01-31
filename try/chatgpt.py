from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Create ChromeOptions object
service = Service(executable_path='C:/chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument(r"--user-data-dir=C:/Users/sujal/AppData/Local/Google/Chrome/User Data")
options.add_argument(r'--profile-directory=Default')
options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# options.add_argument('--headless')  
# # options.add_argument('--disable-gpu') 
# options.add_argument('--window-size=1920,1200')
# # options.add_argument('--no-sandbox')
# options.add_argument('log-level=3')
# options.add_argument('--allow-insecure-localhost')  
driver = webdriver.Chrome(service=service, options=options)

target_url='https://chat.openai.com/c/5f153b76-f2d9-47af-a80c-f1a11256da67/'
driver.get(target_url)
prompt = driver.find_element(By.ID,"prompt-textarea")
time.sleep(5)
prompt.send_keys("COROMANDEL LICI COALINDIA INFY IRFC")
prompt.send_keys(Keys.ENTER)        
time.sleep(10)
text=driver.find_elements(By.XPATH,'//div/div/div[2]/div[2]/div[1]/div/div/p')
print(text[-1].get_attribute('innerHTML'))
