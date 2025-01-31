from bs4 import BeautifulSoup
import pathlib
import shutil
import streamlit as st

# Google Analytics Measurement ID
GA_TRACKING_ID = "G-2MTDPRBPKT"

# Google Analytics script
GA_SCRIPT = f"""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_TRACKING_ID}"></script>
<script id='google_analytics'>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_TRACKING_ID}');
</script>
"""

def inject_ga():
    """Modify Streamlit's index.html to include Google Analytics tracking code."""
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"

    # Read the original index.html
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")

    # Prevent duplicate injections
    if not soup.find(id="google_analytics"):
        # Backup the original file
        bck_index = index_path.with_suffix('.bck')
        if not bck_index.exists():
            shutil.copy(index_path, bck_index)  

        # Insert GA script inside the <head> section
        html = str(soup)
        new_html = html.replace('<head>', '<head>\n' + GA_SCRIPT)

        # Save the modified index.html
        index_path.write_text(new_html)
        print("✅ Google Analytics script injected successfully!")

inject_ga()

