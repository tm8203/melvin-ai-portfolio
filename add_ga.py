from bs4 import BeautifulSoup
import pathlib
import shutil
import streamlit as st

GA_TRACKING_ID = "G-2MTDPRBPKT"  # Make sure this is correct

GA_SCRIPT = f"""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_TRACKING_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_TRACKING_ID}');
</script>
"""

def force_inject_ga():
    """Force injects GA script into Streamlit's index.html on every run"""
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"

    try:
        soup = BeautifulSoup(index_path.read_text(), "html.parser")

        # Always inject after <head> (overwrite if necessary)
        html = str(soup)
        new_html = html.replace("<head>", "<head>\n" + GA_SCRIPT)

        # Backup original index.html if not already backed up
        bck_index = index_path.with_suffix(".bck")
        if not bck_index.exists():
            shutil.copy(index_path, bck_index)

        # Write updated HTML
        index_path.write_text(new_html)
        print("✅ Google Analytics script forcefully injected!")

    except Exception as e:
        print(f"❌ GA Injection Failed: {e}")

# Run the injection every time
force_inject_ga()
