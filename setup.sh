mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[theme]
primaryColor = ‘#CE00FF’
backgroundColor = ‘#000000’
secondaryBackgroundColor = ‘#420056’
textColor= ‘#FFFFFF’
font = ‘sans serif’
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
