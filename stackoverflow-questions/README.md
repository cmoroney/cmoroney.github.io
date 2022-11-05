# Stackoverflow Dataset Analysis
Stackoverflow is the first place any developer goes to in search of solutions for challenges and programming errors that might come up.
SO covers the first page of Google results for most if not all common error codes one might throw at it (trust me).

StackOverflow has published a public dataset containing all forum activity between 2008 and 2022. The largest table, `questions`,
contains over 24 million rows of post content and metadata. 

This dataset is accessible via Google BigQuery here: https://cloud.google.com/bigquery/public-data  

## Problem
Over the years, the userbase has grown quite a bit and as a consequence the number of unanswered questions has grown with it.
![some alt text](/images/stackoverflow_user_growth.png)

The chart below represents the percentage of questions that have not had any posted answers.

![some alt text](/images/unanswered_question_percent.png)

While this paints a picture of the problem, we can dig a little further. Just because someone responds to your question, 
it doesn't necessarily mean your question was answered satisfactorily. 

Using the same table, we can calculate the percentage of questions with an answer that was accepted by 
the original asker, suggesting that their question was answered sufficiently.   

![some alt text](/images/accepted_answer_percent.png)

## Business Case
Unanswered questions could lead to a falling user base for Stack Overflow. This could lead to less advertiser interest, 
and less enterprise business exposure which are two forms of revenue generation at the company currently. 

One potential path forward is to try to curb the number of unanswered questions at the source. 

View the [FIX THIS LINK](https://google.com)   
GitHub Repo: [https://github.com/cmoroney/stackoverflow-questions](https://github.com/cmoroney/stackoverflow-questions)

