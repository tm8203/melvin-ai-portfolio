import pathlib
import streamlit as st
import shutil
from bs4 import BeautifulSoup

# ✅ Google Analytics Tracking ID
GA_TRACKING_ID = "G-2MTDPRBPKT"

# ✅ Google Analytics Injection Script
GA_SCRIPT = f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_TRACKING_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_TRACKING_ID}');
</script>
"""

# ✅ Function to Inject GA into Streamlit's index.html
def inject_ga():
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")
    
    # Avoid Duplicate Injection
    if GA_TRACKING_ID not in str(soup):
        backup_path = index_path.with_suffix(".bkp")
        
        # Create a Backup
        if not backup_path.exists():
            shutil.copy(index_path, backup_path)
        
        # Insert GA Script
        html = str(soup)
        new_html = html.replace("<head>", f"<head>\n{GA_SCRIPT}")
        index_path.write_text(new_html)

# ✅ Inject GA when the script runs
inject_ga()
