# Auto Data Scrape & Analysis
This project was created entirely from curiosity - I am a car hobbyist and spend a lot of time researching automotive 
engineering and have wanted to get my hands on a dataset that has specs for as many cars as possible to perform analysis on.

While I wasn't able to find a public API for this data, I did find a website that has all of the data available to the 
public for free. I scraped this website using BeautifulSoup4, the resulting dataset containing 60+ data points for 42,000
models from 66 makes.

Once the data was pulled, I cleansed it and modeled 0-60mph times using weight to horsepower ratio:

![Image of relationship between 0-60mph and horsepower to weight ratio](https://github.com/cmoroney/auto_data_web_scrape/python/output.png)

Will update this page later with a more in-depth explanation.

GitHub Repo: [https://github.com/cmoroney/auto_data_web_scrape](https://github.com/cmoroney/auto_data_web_scrape)





