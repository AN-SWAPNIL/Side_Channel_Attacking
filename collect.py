import time
import json
import os
import signal
import sys
import random
import traceback
import socket
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import database
from database import Database

WEBSITES = [
    # websites of your choice
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]

TRACES_PER_SITE = 2  # For testing - increase to 1000 for full collection as per spec
FINGERPRINTING_URL = "http://localhost:5000" 
OUTPUT_PATH = "dataset.json"

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)

""" Signal handler to ensure data is saved before quitting. """
def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


"""
Some helper functions to make your life easier.
"""

def is_server_running(host='127.0.0.1', port=5000):
    """Check if the Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def setup_webdriver():
    """Set up the Selenium WebDriver with Chrome options."""
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def retrieve_traces_from_backend(driver):
    """Retrieve traces from the backend API."""
    traces = driver.execute_script("""
        return fetch('/api/get_traces')
            .then(response => response.ok ? response.json() : {traces: []})
            .then(data => data.traces || [])
            .catch(() => []);
    """)
    
    count = len(traces) if traces else 0
    print(f"  - Retrieved {count} traces from backend API" if count else "  - No traces found in backend storage")
    return traces or []

def clear_trace_results(driver, wait):
    """Clear all results from the backend by pressing the button."""
    clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear all results')]")
    clear_button.click()

    wait.until(EC.text_to_be_present_in_element(
        (By.XPATH, "//div[@role='alert']"), "Cleared"))
    
def is_collection_complete():
    """Check if target number of traces have been collected."""
    current_counts = database.db.get_traces_collected()
    remaining_counts = {website: max(0, TRACES_PER_SITE - count) 
                      for website, count in current_counts.items()}
    return sum(remaining_counts.values()) == 0

"""
Your implementation starts here.
"""

def collect_single_trace(driver, wait, website_url):
    """ Implement the trace collection logic here. 
    1. Open the fingerprinting website
    2. Click the button to collect trace
    3. Open the target website in a new tab
    4. Interact with the target website (scroll, click, search)
    5. Return to the fingerprinting tab and close the target website tab
    6. Wait for the trace to be collected
    7. Return success or failure status
    """
    try:
        print(f"  - Collecting trace for {website_url}")
        
        # Clean up any extra tabs first
        if len(driver.window_handles) > 1:
            print(f"  - Cleaning up: {len(driver.window_handles) - 1} extra tabs found")
            main_window = driver.window_handles[0]
            for handle in driver.window_handles[1:]:
                driver.switch_to.window(handle)
                driver.close()
            driver.switch_to.window(main_window)
        
        # Navigate to fingerprinting page
        driver.get(FINGERPRINTING_URL)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "main")))
        
        # Store the fingerprinting tab handle
        fingerprinting_tab = driver.current_window_handle
        
        # Click the "Collect Trace Data" button
        collect_button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Collect Trace Data')]")
        ))
        collect_button.click()
        
        # Wait a moment for the collection to start
        time.sleep(1)
        
        # Set a start time to track interaction duration
        interaction_start_time = time.time()
        max_interaction_time = 7  # seconds
        
        # Open target website in a new tab
        driver.execute_script(f"window.open('{website_url}', '_blank');")
        
        # Wait for the new tab to open and switch to it
        wait.until(EC.number_of_windows_to_be(2))
        target_window = [window for window in driver.window_handles if window != fingerprinting_tab][0]
        driver.switch_to.window(target_window)
        
        # Wait for target website to load
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # Website-specific interactions based on URL
        should_exit_early = False
        print(f"  - Interacting with website: {website_url}")
        
        try:
            # Website-specific interactions
            if "google.com" in website_url:
                print(f"  - Google homepage interaction - clicking buttons instead of searching")
                
                # Accept cookies dialog if present (helps avoid CAPTCHA)
                try:
                    accept_buttons = driver.find_elements(By.XPATH, 
                        "//button[contains(text(), 'Accept') or contains(text(), 'I agree') or contains(text(), 'Accept all')]")
                    if accept_buttons and len(accept_buttons) > 0:
                        accept_buttons[0].click()
                        time.sleep(0.8)
                except:
                    pass
                
                # First, type in the search box but don't submit
                try:
                    search_box = driver.find_element(By.NAME, "q")
                    sample_text = random.choice(["weather", "news", "translate", "maps", "images"])
                    
                    # Type some text with human-like timing
                    for char in sample_text:
                        search_box.send_keys(char)
                        time.sleep(random.uniform(0.1, 0.3))
                    
                    time.sleep(0.5)  # Pause after typing
                    print(f"  - Typed '{sample_text}' in search box")
                except Exception as e:
                    print(f"  - Could not interact with search box: {str(e)}")
                
                # Now click on different Google buttons/links instead of searching
                clickable_elements = []
                
                try:
                    # Try to find "I'm Feeling Lucky" button
                    lucky_buttons = driver.find_elements(By.XPATH, 
                        "//input[@value='I\\'m Feeling Lucky'] | //button[contains(text(), 'Feeling Lucky')] | //input[@name='btnI']")
                    if lucky_buttons:
                        clickable_elements.extend(lucky_buttons)
                        print("  - Found 'I'm Feeling Lucky' button")
                except:
                    pass
                
                try:
                    # Try to find Google apps button (grid icon)
                    apps_buttons = driver.find_elements(By.XPATH, 
                        "//a[@title='Google apps'] | //a[@aria-label='Google apps'] | //*[@data-ved and contains(@aria-label, 'app')]")
                    if apps_buttons:
                        clickable_elements.extend(apps_buttons)
                        print("  - Found Google apps button")
                except:
                    pass
                
                try:
                    # Try to find Gmail link
                    gmail_links = driver.find_elements(By.XPATH, 
                        "//a[contains(text(), 'Gmail')] | //a[@href*='gmail']")
                    if gmail_links:
                        clickable_elements.extend(gmail_links)
                        print("  - Found Gmail link")
                except:
                    pass
                
                try:
                    # Try to find Images link
                    images_links = driver.find_elements(By.XPATH, 
                        "//a[contains(text(), 'Images')] | //a[@href*='images.google']")
                    if images_links:
                        clickable_elements.extend(images_links)
                        print("  - Found Images link")
                except:
                    pass
                
                try:
                    # Try to find language/region links at bottom
                    lang_links = driver.find_elements(By.XPATH, 
                        "//a[contains(text(), 'বাংলা')] | //a[contains(text(), 'English')] | //footer//a[contains(@href, 'setprefs')]")
                    if lang_links:
                        clickable_elements.extend(lang_links[:2])  # Only take first 2
                        print("  - Found language/region links")
                except:
                    pass
                
                try:
                    # Try to find About, Business, etc. links at bottom
                    footer_links = driver.find_elements(By.XPATH, 
                        "//a[contains(text(), 'About')] | //a[contains(text(), 'Business')] | //a[contains(text(), 'How Search works')] | //a[contains(text(), 'Advertising')]")
                    if footer_links:
                        clickable_elements.extend(footer_links[:3])  # Only take first 3
                        print("  - Found footer links")
                except:
                    pass
                
                try:
                    # Try to find Settings link
                    settings_links = driver.find_elements(By.XPATH, 
                        "//a[contains(text(), 'Settings')] | //button[contains(text(), 'Settings')] | //*[@data-ved and contains(text(), 'Settings')]")
                    if settings_links:
                        clickable_elements.extend(settings_links)
                        print("  - Found Settings link")
                except:
                    pass
                
                # If we found clickable elements, click one of them
                if clickable_elements:
                    element_to_click = random.choice(clickable_elements)
                    try:
                        # Get element info for logging
                        element_text = element_to_click.text or element_to_click.get_attribute('aria-label') or element_to_click.get_attribute('title') or "Unknown"
                        print(f"  - Clicking on: {element_text[:30]}")
                        
                        # Scroll to element first
                        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element_to_click)
                        time.sleep(0.8)
                        
                        # Try regular click first
                        try:
                            element_to_click.click()
                            print("  - Successfully clicked element")
                        except:
                            # If regular click fails, try JavaScript click
                            print("  - Regular click failed, trying JavaScript click")
                            driver.execute_script("arguments[0].click();", element_to_click)
                        
                        time.sleep(1.5)  # Wait for any page changes
                        
                        # Check if we're still on Google homepage or navigated somewhere
                        current_url = driver.current_url
                        if "google.com" in current_url and current_url != website_url:
                            print(f"  - Navigated to: {current_url[:60]}...")
                            
                            # Do some scrolling on the new page
                            scroll_positions = [200, 400, 600]
                            for pos in scroll_positions:
                                driver.execute_script(f"window.scrollTo(0, {pos});")
                                time.sleep(random.uniform(0.7, 1.0))
                                
                                if time.time() - interaction_start_time > max_interaction_time * 0.8:
                                    break
                        else:
                            print("  - Stayed on homepage, doing additional scrolling")
                            # Just scroll on the current page
                            scroll_positions = [200, 450, 650]
                            for pos in scroll_positions:
                                driver.execute_script(f"window.scrollTo(0, {pos});")
                                time.sleep(random.uniform(0.7, 1.0))
                                
                                if time.time() - interaction_start_time > max_interaction_time * 0.8:
                                    break
                                    
                    except Exception as click_error:
                        print(f"  - Error clicking element: {str(click_error)}")
                        # Fallback to just scrolling
                        for i in range(3):
                            scroll_pos = (i+1) * 250
                            driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
                            time.sleep(0.8)
                else:
                    print("  - No clickable elements found, performing fallback scrolling")
                    # Fallback scrolling if no clickable elements found
                    for i in range(4):
                        scroll_pos = (i+1) * 200
                        driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
                        time.sleep(random.uniform(0.7, 1.0))
                        
                        if time.time() - interaction_start_time > max_interaction_time * 0.8:
                            break
            
            elif "moodle" in website_url:
                print(f"  - Moodle interaction with scrolling and clicking random links")
                
                # First scroll through the page with slower, more human-like scrolling
                scroll_positions = [150, 300, 450]
                random.shuffle(scroll_positions)
                for pos in scroll_positions:
                    driver.execute_script(f"window.scrollTo(0, {pos});")
                    # Longer pauses between scrolls (0.7-1.0 seconds)
                    time.sleep(random.uniform(0.7, 1.0))
                    
                    # Check if we're approaching the time limit
                    if time.time() - interaction_start_time > max_interaction_time * 0.7:
                        print("  - Approaching time limit, skipping link click in Moodle")
                        # Skip the link clicking part if we're running out of time
                        should_exit_early = True
                        break
                
                # Try to click a link - different one each time
                # if not should_exit_early:
                    try:
                        # Get all links
                        all_links = driver.find_elements(By.TAG_NAME, "a")
                        
                        # Filter to visible links with text
                        visible_links = []
                        for link in all_links:
                            if link.is_displayed() and link.text and len(link.text.strip()) > 0:
                                visible_links.append(link)
                        
                        if visible_links:
                            print(f"  - Found {len(visible_links)} visible links to click")
                            random.shuffle(visible_links)
                            random_index = random.randint(0, len(visible_links)-1)
                            
                            # Select a random link to click
                            link_to_click = visible_links[random_index]
                            print(f"  - Clicking on link: {link_to_click.text[:20]}... (index {random_index})")
                            
                            # Scroll to the link in a more human-like way
                            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", link_to_click)
                            time.sleep(0.8)  # Longer pause before clicking
                            
                            # Click the link
                            link_to_click.click()
                            time.sleep(1.2)  # Longer wait for page load
                            
                            # Do some scrolling on the new page - slower and more varied
                            scroll_amounts = [200, 350, 500]
                            random.shuffle(scroll_amounts)
                            for amount in scroll_amounts:
                                driver.execute_script(f"window.scrollBy(0, {amount});")
                                time.sleep(random.uniform(0.8, 1.1))  # Longer, varied pauses
                                
                                # Check time limit
                                if time.time() - interaction_start_time > max_interaction_time * 0.9:
                                    print("  - Approaching time limit, ending Moodle page scrolling early")
                                    should_exit_early = True
                                    break
                            
                            # Skip "back" if we're out of time
                            if not should_exit_early:
                                # Go back to previous page
                                driver.back()
                                time.sleep(1.0)  # Longer wait after going back
                    except Exception as e:
                        print(f"  - Error clicking Moodle link: {str(e)}")
                        # If clicking fails, do additional scrolling
                        for pos in [300, 500, 700]:
                            driver.execute_script(f"window.scrollTo(0, {pos});")
                            time.sleep(0.8)  # Slower scrolling
            
            elif "prothomalo.com" in website_url:
                print(f"  - Prothom Alo interaction with scrolling and single tab click")
                
                # Initial slower scroll with varied pauses
                scroll_positions = [150, 350, 550]
                random.shuffle(scroll_positions)
                for pos in scroll_positions:
                    driver.execute_script(f"window.scrollTo(0, {pos});")
                    # Human-like varied pauses (0.8-1.2 seconds)
                    time.sleep(random.uniform(0.8, 1.2))
                    
                    # Check if we're approaching the time limit
                    if time.time() - interaction_start_time > max_interaction_time * 0.6:
                        print("  - Approaching time limit, skipping tab click in Prothom Alo")
                        # Skip the tab clicking part if we're running out of time
                        should_exit_early = True
                        break
                
                # Try to click one tab/link
                # if not should_exit_early:
                    try:
                        # Focus specifically on news articles first
                        news_articles = driver.find_elements(By.CSS_SELECTOR, 
                            ".news-item a, article a, .headline a, .title a, .card a, .story a, .news a, " +
                            "h1 a, h2 a, h3 a, .item-headline a")
                        
                        # Filter to visible news elements with content
                        clickable_news = []
                        for elem in news_articles:
                            if elem.is_displayed() and elem.text and len(elem.text.strip()) > 3:
                                clickable_news.append(elem)
                        
                        # If no news articles found, fall back to navigation elements
                        if not clickable_news:
                            print("  - No news articles found, looking for other clickable elements")
                            nav_tabs = driver.find_elements(By.CSS_SELECTOR, 
                                "header a, nav a, .menu a, .nav a, .navigation a")
                            
                            if nav_tabs and len(nav_tabs) > 1:
                                # Filter to visible elements with content
                                for elem in nav_tabs[1:5]:  # Skip first which is often logo
                                    if elem.is_displayed() and elem.text and len(elem.text.strip()) > 3:
                                        clickable_news.append(elem)
                        
                        if clickable_news:
                            # Select one news element to click - prioritize articles
                            element_to_click = random.choice(clickable_news)
                            print(f"  - Clicking on news: {element_to_click.text[:20]}...")
                            
                            # Scroll to element with smooth behavior
                            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element_to_click)
                            time.sleep(1.0)  # Longer pause before clicking
                            
                            # Click it
                            element_to_click.click()
                            time.sleep(1.2)  # Longer wait for page load
                            
                            # Scroll on the new page with human-like pauses - more thorough for news articles
                            scroll_positions = [250, 500, 750, 1000]
                            for pos in scroll_positions:
                                driver.execute_script(f"window.scrollTo(0, {pos});")
                                time.sleep(random.uniform(0.8, 1.2))  # Slower, varied scrolling
                                
                                # Check time limit
                                if time.time() - interaction_start_time > max_interaction_time * 0.9:
                                    print("  - Approaching time limit, ending Prothom Alo article scrolling early")
                                    should_exit_early = True
                                    break
                                    
                            # Skip additional scrolling if we're out of time
                            if not should_exit_early:
                                # Additional scrolling if it's a longer article
                                try:
                                    # Try to find the main article content
                                    article_body = driver.find_elements(By.CSS_SELECTOR, 
                                        "article, .article-body, .story-content, .news-content, .content-body, .body-text")
                                    
                                    if article_body and len(article_body[0].text) > 300:
                                        # For longer articles, scroll deeper with natural pauses between scrolls
                                        print("  - Found longer article, scrolling through content")
                                        deeper_positions = [1200, 1500, 1800, 2100]
                                        for pos in deeper_positions:
                                            driver.execute_script(f"window.scrollTo(0, {pos});")
                                            time.sleep(random.uniform(0.9, 1.2))  # Slower scrolling
                                            
                                            # Check time limit again
                                            if time.time() - interaction_start_time > max_interaction_time * 0.95:
                                                print("  - Approaching time limit, ending deep article scrolling")
                                                break
                                except Exception as e:
                                    print(f"  - Error during article scrolling: {str(e)}")
                                    pass
                    except Exception as e:
                        print(f"  - Error navigating Prothom Alo: {str(e)}")
                        # If clicking fails, just do additional scrolling
                        for pos in [200, 400, 600, 800]:
                            driver.execute_script(f"window.scrollTo(0, {pos});")
                            time.sleep(0.9)  # Slower scrolling
                            
                            # Check time limit
                            if time.time() - interaction_start_time > max_interaction_time * 0.9:
                                break
            
            else:
                # Default behavior for other websites - just scroll
                for i in range(3):  # Minimal scrolling for speed
                    scroll_amount = 300
                    driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                    time.sleep(0.2)  # Quick pause
                    
                    # Check if we're approaching the time limit
                    if time.time() - interaction_start_time > max_interaction_time:
                        print("  - Approaching time limit, ending interaction early")
                        should_exit_early = True
                        break
        
        except Exception as e:
            print(f"  - Error during website interaction: {str(e)}")
            # Fallback scrolling
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(0.5)
        
        # Report interaction time
        interaction_time = time.time() - interaction_start_time
        print(f"  - Website interaction completed in {interaction_time:.2f} seconds")
        
        # Switch back to fingerprinting tab (but keep target website open)
        driver.switch_to.window(fingerprinting_tab)
        
        # Wait for trace collection to complete (look for status message)
        try:
            wait.until(EC.text_to_be_present_in_element(
                (By.XPATH, "//div[@role='alert']"), "Traced"))
            time.sleep(1)  # Reduced wait time for efficiency
        except:
            print("    - Warning: No status alert found, continuing...")
        
        # Now close the target website tab after trace is complete
        for handle in driver.window_handles:
            if handle != fingerprinting_tab:
                driver.switch_to.window(handle)
                driver.close()
        driver.switch_to.window(fingerprinting_tab)
        
        print(f"    - Successfully collected trace for {website_url}")
        return True
        
    except Exception as e:
        print(f"    - Error collecting trace for {website_url}: {str(e)}")
        return False

def collect_fingerprints(driver, target_counts=None):
    """ Implement the main logic to collect fingerprints.
    1. Calculate the number of traces remaining for each website
    2. Open the fingerprinting website
    3. Collect traces for each website until the target number is reached
    4. Save the traces to the database
    5. Return the total number of new traces collected
    """
    wait = WebDriverWait(driver, 30)  # Set up a wait with longer timeout for stability
    
    # Navigate to fingerprinting website initially
    print(f"Navigating to fingerprinting website: {FINGERPRINTING_URL}")
    driver.get(FINGERPRINTING_URL)
    wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Collect Trace Data')]")))
    
    # Get current collection counts
    current_counts = database.db.get_traces_collected()
    print(f"Current trace counts: {current_counts}")
    
    # Filter to only include original websites
    website_counts = {}
    for website in WEBSITES:
        # Get count for the original website
        count = current_counts.get(website, 0)
        website_counts[website] = count
    
    # Calculate remaining traces to collect for each website
    if target_counts is None:
        remaining_counts = {website: max(0, TRACES_PER_SITE - website_counts.get(website, 0)) 
                          for website in WEBSITES}
    else:
        remaining_counts = target_counts
    
    print(f"Remaining traces to collect: {remaining_counts}")
    
    total_new_traces = 0
    
    # Loop through websites until all targets are met
    while sum(remaining_counts.values()) > 0:
        # Get website with most remaining traces
        website = max(remaining_counts.items(), key=lambda x: x[1])[0]
        if remaining_counts[website] <= 0:
            # If all websites are done, break
            break
            
        print(f"\nCollecting trace for {website}. Remaining: {remaining_counts[website]}")
        
        
        
        # Collect a single trace (your enhanced version handles trace_id parameter)
        trace_success = collect_single_trace(driver, wait, website)
        
        if trace_success:
            # After successful collection, retrieve the trace data
            print("  - Getting trace data from browser")
            
            try:
                # Wait a moment to ensure data is fully processed
                time.sleep(1)
                
                # Ensure browser is in clean state (only one tab open)
                cleanup_browser(driver)
                
                # Retrieve trace data from backend
                api_traces = retrieve_traces_from_backend(driver)
                
                # If we got traces, save the latest one
                if api_traces and len(api_traces) > 0:
                    latest_trace = api_traces[-1]  # Get the most recently added trace
                    website_index = WEBSITES.index(website)
                    
                    # Extract the trace data - handle different data structures
                    trace_data = None
                    if isinstance(latest_trace, dict):
                        if 'data' in latest_trace:
                            trace_data = latest_trace['data']
                        elif 'trace_data' in latest_trace:
                            trace_data = latest_trace['trace_data']
                    
                    if not trace_data and isinstance(latest_trace, list):
                        # Try to get raw data if it's already a list
                        trace_data = latest_trace
                        
                    # Save trace to database with the original website name (not unique identifier)
                    if trace_data and database.db.save_trace(website, website_index, trace_data):
                        # Update our tracking of remaining traces for the original website
                        remaining_counts[website] -= 1
                        total_new_traces += 1
                        print(f"  - Saved trace {total_new_traces} for {website}. Remaining: {remaining_counts[website]}")
                        
                        # Export JSON after each save to update dataset.json incrementally
                        database.db.export_to_json(OUTPUT_PATH)
                        print(f"  - Updated {OUTPUT_PATH} with new data")
                        
                        # Update collection stats properly through the database interface
                        # (The save_trace method should handle this automatically)
                    else:
                        print(f"  - Failed to save trace to database")
                else:
                    # If API retrieval fails, try a more direct approach - create a manual trace
                    print("  - No trace data retrieved from API")
                    raise ValueError("No trace data found after collection")
                    
                # Clear results after every successful trace collection to prevent the array from growing too large
                print("  - Clearing trace results from UI")
                time.sleep(1)  # Ensure UI is ready for clearing
                clear_trace_results(driver, wait)
                # Additional brief pause after clearing for stability
                    
            except Exception as e:
                print(f"  - Error processing trace: {str(e)}")
                traceback.print_exc()
                
            # Brief pause between collections (reduced for efficiency)
            # time.sleep(2)  
        else:
            print("  - Trace collection failed, retrying...")
            time.sleep(5)  # Longer pause after failure
            
            # Try to refresh the page if collection failed
            try:
                driver.refresh()
                wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Collect Trace Data')]")))
            except:
                print("  - Refresh failed, continuing anyway")
    
    print(f"\nFinished collection. Total new traces: {total_new_traces}")
    return total_new_traces

def cleanup_browser(driver):
    """Ensure browser is in clean state with only one tab open"""
    try:
        if len(driver.window_handles) > 1:
            print(f"  - Browser cleanup: {len(driver.window_handles) - 1} extra tabs found")
            main_window = driver.window_handles[0]
            for handle in driver.window_handles[1:]:
                driver.switch_to.window(handle)
                driver.close()
            driver.switch_to.window(main_window)
    except Exception as e:
        print(f"  - Warning: Browser cleanup failed: {str(e)}")

def main():
    """ Implement the main function to start the collection process.
    1. Check if the Flask server is running
    2. Initialize the database
    3. Set up the WebDriver
    4. Start the collection process, continuing until the target number of traces is reached
    5. Handle any exceptions and ensure the WebDriver is closed at the end
    6. Export the collected data to a JSON file
    7. Retry if the collection is not complete
    """
    print("Starting Website Fingerprinting Data Collection...")
    
    # Check if Flask server is running
    if not is_server_running():
        print("ERROR: Flask server is not running!")
        print("Please start the server first by running: python app.py")
        return
    
    print("✓ Flask server is running")
    
    # Initialize database
    try:
        database.db.init_database()
        print("✓ Database initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize database: {str(e)}")
        return
    
    driver = None
    retry_count = 0
    max_retries = 3
    
    while not is_collection_complete() and retry_count < max_retries:
        try:
            print(f"\n=== Collection Attempt {retry_count + 1} ===")
            
            # Set up WebDriver
            print("Setting up browser...")
            driver = setup_webdriver()
            print("✓ Browser setup complete")
            
            # Start collection process
            new_traces = collect_fingerprints(driver)
            print(f"✓ Collection round complete. New traces: {new_traces}")
            
            # Check if collection is complete
            current_counts = database.db.get_traces_collected()
            print(f"Current status: {current_counts}")
            
            if is_collection_complete():
                print("🎉 Target collection complete!")
                break
            else:
                print("Collection not yet complete, will retry...")
                retry_count += 1
                
        except KeyboardInterrupt:
            print("\n⚠️  Collection interrupted by user")
            break
            
        except Exception as e:
            print(f"❌ Error during collection: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            retry_count += 1
            
        finally:
            # Always close the driver
            if driver:
                try:
                    driver.quit()
                    print("✓ Browser closed")
                except:
                    pass
                driver = None
            
            # Small delay before retry
            if retry_count < max_retries and not is_collection_complete():
                print(f"Waiting before retry attempt {retry_count + 1}...")
                time.sleep(5)
    
    # Export collected data
    try:
        print("\nExporting data to JSON...")
        database.db.export_to_json(OUTPUT_PATH)
        print(f"✓ Data exported to {OUTPUT_PATH}")
    except Exception as e:
        print(f"❌ Error exporting data: {str(e)}")
    
    final_counts = database.db.get_traces_collected()
    print(f"Final status: {final_counts}")
    
    total_collected = sum(final_counts.values())
    target_total = len(WEBSITES) * TRACES_PER_SITE
    
    print(f"\n=== Final Collection Summary ===")
    print(f"Total traces collected: {total_collected}/{target_total}")
    for website, count in final_counts.items():
        print(f"  {website}: {count}/{TRACES_PER_SITE}")
    
    if is_collection_complete():
        print("🎉 Collection target achieved!")
    else:
        print("⚠️  Collection target not fully achieved")
        if retry_count >= max_retries:
            print(f"❌ Maximum retries ({max_retries}) reached")

if __name__ == "__main__":
    main()
