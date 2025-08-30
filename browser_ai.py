from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# --- Setup WebDriver ---
driver = webdriver.Chrome()  # Make sure chromedriver is installed and in PATH

# Open weather website
driver.get("https://www.weather.com/")
time.sleep(3)

# Search for a city
search_box = driver.find_element(By.ID, "LocationSearch_input")
city_name = "Bangalore"
search_box.send_keys(city_name)
search_box.send_keys(Keys.RETURN)
time.sleep(3)

# Click on first suggestion (if needed)
try:
    first_suggestion = driver.find_element(By.CSS_SELECTOR, "button.styles__item__3sdr8")
    first_suggestion.click()
    time.sleep(2)
except:
    pass

# Get current temperature
try:
    temperature = driver.find_element(By.CSS_SELECTOR, "span.CurrentConditions--tempValue--3KcTQ")
    condition = driver.find_element(By.CSS_SELECTOR, "div.CurrentConditions--phraseValue--2xXSr")
    print(f"Weather in {city_name}: {temperature.text}, {condition.text}")
except:
    print("Could not fetch weather data.")

# Close browser
driver.quit()
