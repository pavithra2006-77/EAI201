from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# 1️⃣ Setup Chrome
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

# 2️⃣ Open Wikipedia
driver.get("https://en.wikipedia.org")
time.sleep(3)

# 3️⃣ Search for topic
topic = "Newton's Second Law"
search_box = driver.find_element(By.NAME, "search")
search_box.clear()
search_box.send_keys(topic)
search_box.send_keys(Keys.RETURN)
time.sleep(3)

# 4️⃣ Open first search result if needed
if "search" in driver.current_url:
    try:
        first_result = driver.find_element(By.XPATH, "//div[@class='mw-search-result-heading']/a")
        first_result.click()
        time.sleep(3)
    except:
        print("No search results found, continuing with current page.")

# 5️⃣ Extract descriptive content
try:
    content_div = driver.find_element(By.CLASS_NAME, "mw-parser-output")
    paragraphs = content_div.find_elements(By.TAG_NAME, "p")
    
    # Collect first 3 meaningful paragraphs
    descriptive_text = ""
    count = 0
    for p in paragraphs:
        text = p.text.strip()
        if len(text) > 50:
            descriptive_text += text + "\n\n"
            count += 1
        if count == 3:
            break

    if descriptive_text:
        print(f"**Topic:** {topic}\n")
        print("**Descriptive Summary:**\n")
        print(descriptive_text)
    else:
        print("No meaningful content found on the page.")

except Exception as e:
    print(f"Could not extract content. Error: {e}")

time.sleep(5)
driver.quit()
