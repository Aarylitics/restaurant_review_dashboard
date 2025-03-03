import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# Streamlit UI Configuration
st.set_page_config(page_title="Restaurant Review Dashboard", layout="wide")
st.sidebar.title('Restaurant Review Dashboard')
st.sidebar.write("Enter a Google Maps URL to scrape restaurant reviews.")

# User input for Google Maps URL
url = st.sidebar.text_input("Google Maps URL", "")

# Selenium WebDriver Setup
def setup_driver():
    options = Options()
    options.add_argument("--headless")  # Run in headless mode for Streamlit
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service("chromedriver")
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# Function to extract restaurant name and reviews
def scrape_reviews(url):
    driver = setup_driver()
    driver.get(url)
    time.sleep(3)
    
    try:
        # Extract restaurant name
        name = driver.find_element(By.TAG_NAME, "h1").text
        
        # Click on reviews button
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'reviews')]"))
        ).click()
        time.sleep(3)

        # Scroll to load more reviews
        scrollable_div = driver.find_element(By.CLASS_NAME, "m6QErb")
        for _ in range(5):  # Adjust range for more reviews
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
            time.sleep(2)
        
        # Extract review elements
        reviews = driver.find_elements(By.CLASS_NAME, "wiI7pd")
        ratings = driver.find_elements(By.CLASS_NAME, "kvMYJc")
        
        review_data = []
        for review, rating in zip(reviews, ratings):
            review_data.append({
                "review": review.text,
                "rating": rating.get_attribute("aria-label")
            })
        
        driver.quit()
        return name, pd.DataFrame(review_data)
    
    except Exception as e:
        driver.quit()
        return None, f"Error: {str(e)}"

# Run scraping and display results
if url:
    restaurant_name, review_df = scrape_reviews(url)
    
    if isinstance(review_df, str):
        st.error(review_df)
    else:
        st.sidebar.write(f"**Restaurant Name**: {restaurant_name}")
        st.write("### Customer Reviews")
        st.dataframe(review_df)
        
        # Show rating distribution
        st.write("### Rating Distribution")
        review_df["rating"] = review_df["rating"].str.extract(r'(\d\.\d)')[0].astype(float)
        st.bar_chart(review_df["rating"].value_counts().sort_index())
