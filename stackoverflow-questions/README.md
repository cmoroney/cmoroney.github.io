# Stackoverflow Dataset Analysis
Stackoverflow is the first place any developer goes to in search of solutions for challenges and programming errors that might come up.
SO covers the first page of Google results for most if not all common error codes one might throw at it (trust me).

StackOverflow has published a public dataset containing all forum activity between 2008 and 2022. The largest table, `questions`,
contains over 24 million rows of post content and metadata. 

This dataset is accessible via Google BigQuery here: https://cloud.google.com/bigquery/public-data  

## Problem
Over the years, the userbase has grown quite a bit and as a consequence the number of unanswered questions has grown with it.
![data_visualization](/images/stackoverflow_user_growth.png)

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

One potential path forward is to identify questions as unanswerable before they are posted. Classifying questions as unanswerable 
before they are posted may give internal teams at the company an opportunity to intervene - show the user a "answerability" 
score, offer suggestions to improve the body of their question, suggest better tags, etc. 

## Data Prep
I will be querying the public dataset from a Jupyter notebook and storing the result in a table in my account for future use. 

### Question Features
The `questions` table has a lot of information that you can also see on the post's corresponding webpage. Features created  
from fields in this table:
- Creation date - year, day of week, hour 
- Body - this contains html of the post content. HTML and `code` tags will be removed to generate text features
- Title - character count, question words included
- Total `<code>` snippet count
- `<code>` snippet length
- List tags included in body (boolean)

### User Features
The public dataset includes a `users` table that could provide additional information about the person asking the question. 

There is also a table containing badge data. Badges are earned by users for completing different accomplishments. For example, 
when you ask a question and accept an answer, you will receive a "Scholar" badge. 

- Total question badges
- Total answer badges
- Total "other" badges
- Total bronze, silver, and gold tag badges
- Number of questions asked
- Cumulative post score
- User tenure - calculated as a day difference between sign-up and the date which the question is asked
- User profile score

### Response variable
I will use questions that no not have any accepted answers as a definition of an "unanswered" question. This provides a 
more accurate response and provides a more balanced dataset to classify - 49% of all questions have no accepted answer 
vs 14% having no answer at all.

### Completed SQL query:
```sql
WITH stackoverflow_questions AS (
        SELECT *
        FROM bigquery-public-data.stackoverflow.posts_questions
        TABLESAMPLE SYSTEM (20 PERCENT)
    ),
    -- WITH
    badges AS (
    -- More info at https://stackoverflow.com/help/badges
      SELECT
        date AS earned_date,
        id,
        user_id,
        name,
        CASE
          WHEN LOWER(name) like '%question%' OR
            name IN ('Altruist', 'Benefactor', 'Curious', 'Inquisitive', 'Socratic', 'Investor', 'Promoter', 'Scholar', 'Student')
          THEN 'question'
          WHEN LOWER(name) like '%answer%' or
            name IN ('Enlightened', 'Explainer', 'Refiner', 'Illuminator', 'Generalist', 'Guru', 'Lifejacket', 'Lifeboat', 'Populist', 'Revival', 'Necromancer', 'Self-Learner', 'Teacher', 'Tenacious', 'Unsung Hero')
          THEN 'answer'
          WHEN tag_based AND class = 1 THEN 'gold_tag'
          WHEN tag_based AND class = 2 THEN 'silver_tag'
          WHEN tag_based AND class = 3 THEN 'bronze_tag'
          ELSE 'other_badge'
        END AS badge_type
      FROM bigquery-public-data.stackoverflow.badges
      ORDER BY 2
    ),`

    calc_features AS (
    SELECT
      q.id,
      q.creation_date,
      q.owner_user_id,
      q.body,
      q.title,
      regexp_replace(
        regexp_replace(
                    regexp_replace(q.body, r'''<code>(.|\s)*</code>''', ''), r'''<([a-zA-Z\s]|/[a-zA-Z\s])*>''', ''
                ), '''<[^>]+>''', ''''''
            ) as body_text,

      -- Date features
      EXTRACT(year FROM q.creation_date) AS year,
      EXTRACT(dayofweek FROM q.creation_date) AS dow,
      EXTRACT(hour from q.creation_date) AS hour,

      -- Title / body features
       array_to_string(regexp_extract_all(body, r'''<code>([^<]+)<\/code>'''), '''\n''') AS code_text,
      array_length(regexp_extract_all(body, r'''<code>([^<]+)<\/code>''')) AS total_code_snippets,
      length(array_to_string(regexp_extract_all(body, r'''<code>([^<]+)<\/code>'''), '''\n''')) AS code_length,

      length(replace(q.title, ''' ''', '')) AS title_character_count,
      array_length(regexp_extract_all(trim(q.title), ''' ''')) + 1 AS title_word_count,
      array_length(split(trim(regexp_replace(tags, r'''\|''', ',')))) AS total_tags,
      regexp_contains(title, '''^\\b[A-Z][a-z]*\\b''') AS title_init_title_case,
      regexp_contains(title, r'''\\?$''') AS title_term_question,
      regexp_contains(lower(title), '^who|what|when|where|why|how .*$') AS title_init_wh,
      body like '%<ul>%<li>%</li>%</ul>%' AS body_contains_list,

      -- User profile features
      u.about_me IS NOT NULL AS has_about_me,
      u.profile_image_url IS NOT NULL as has_profile_image,
      u.website_url IS NOT NULL AS has_website_url,
      q.score,

      -- User history features
      DATE_DIFF(q.creation_date, u.creation_date, DAY) AS user_tenure,
      COALESCE(
        SUM(q.score) OVER (PARTITION BY q.owner_user_id
          ORDER BY q.creation_date
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
       , 0) AS cummulative_post_score,
      rank() OVER(PARTITION BY q.owner_user_id ORDER BY q.creation_date) AS questions_asked,
      COUNT(CASE WHEN b.badge_type = 'question' THEN b.id END) AS question_badges,
      COUNT(CASE WHEN b.badge_type = 'answer' THEN b.id END) AS answer_badges,
      COUNT(CASE WHEN b.badge_type = 'other_badge' THEN b.id END) AS other_badges,
      COUNT(CASE WHEN b.badge_type = 'bronze_tag' THEN b.id END) AS bronze_tag_badges,
      COUNT(CASE WHEN b.badge_type = 'silver_tag' THEN b.id END) AS silver_tag_badges,
      COUNT(CASE WHEN b.badge_type = 'gold_tag' THEN b.id END) AS gold_tag_badges,

      -- Two options for response variable
      SUM(answer_count) > 0 AS answer_boolean,
      accepted_answer_id IS NOT NULL AS accepted_answer_boolean

    FROM stackoverflow_questions q
    LEFT JOIN badges b
      ON q.owner_user_id = b.user_id
      AND b.earned_date < q.creation_date
    LEFT JOIN `bigquery-public-data.stackoverflow.users` u
      ON q.owner_user_id = u.id
    WHERE q.owner_user_id IS NOT NULL
    GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,34
    )

    SELECT
      id,
      -- Text features
      body_text,
      title AS title_text,

      -- Categorical features
      year AS year_cat,
      dow AS dow_cat,
      hour AS hour_cat,
      title_init_title_case AS title_case_cat,
      title_term_question AS title_question_cat,
      title_init_wh AS title_w_cat,
      body_contains_list AS body_list_cat,
      has_about_me AS about_me_cat,
      has_profile_image AS profile_image_cat,
      has_website_url AS website_url_cat,

      -- Numeric features
      length(body_text) AS body_length_num,
      total_code_snippets AS code_snippets_num,
      code_length AS code_length_num,
      code_length / nullif(length(body_text), 0) AS code_to_words_num,
      title_character_count AS title_length_num,
      title_word_count AS title_wordcount_num,
      total_tags AS tag_count_num,
      user_tenure AS user_tenure_days_num,
      cummulative_post_score AS cumm_post_score_num,
      questions_asked AS ques_asked_num,
      question_badges AS ques_badges_num,
      answer_badges AS answer_badges_num,
      other_badges AS other_badges_num,
      bronze_tag_badges AS bronze_badges_num,
      silver_tag_badges AS silver_badges_num,
      gold_tag_badges AS gold_badges_num,

      -- Response variable
      answer_boolean,
      accepted_answer_boolean

    FROM calc_features
```

## Additional data prep in Python
Once the data is extracted from BigQuery, we can start to manipulate it in Python to enrich the dataset with more features.

### Compress integer columns 
Since this query outputs a fairly large dataset (~20 million rows x 37 columns), we will need to make some adjustments along the 
way to assure that we can work with it in-memory. First we will compress the `int` columns as Google BigQuery assigns `int64`
by default. Since the numeric data in the dataset are smaller integers, we can take advantage of `int8`, `int16`, and 
`int32` data types to shrink the size of our training set. 

Function:
```python
def compress_int_columns(data):
    int8_cols = []
    int16_cols = []
    for i in df.columns[data.dtypes == 'int64']:
        if i != "Unnamed: 0":
            col_max = data[i].max()
            # col_min = data[i].min()

            if col_max < 32767:
                if col_max < 127:
                    int8_cols.append(i)
                else:
                    int16_cols.append(i)

    for col in int8_cols:
        data[col] = data[col].astype(np.int8)

    for col in int16_cols:
        data[col] = data[col].astype(np.int16)

    return data
```
Then we can use this to compress any `int64` columns by writing 

```python
df = compress_int_columns(df)
```

## Simple text features
We can add a few additional text features by writing a few one-line functions:

Count total characters:
```python
def count_chars(str):
    return len(str)

def word_count(str):
    return len(str.split())
    
def count_unique_words(str):
        return len(set(str.split()))
```

We can also use more advanced methods from the `textstatistics` and `SpaCy` packages:

```python
def syllables_count(text):
        return textstatistics().syllable_count(text)

# Count total number of sentences and difficult words 
def add_spacy_features(data):
        sentences_out = []
        diff_words_out = []
        for chunk in np.array_split(data, 10):
            nlp = spacy.load('en_core_web_sm')
            docs = list(nlp.pipe(chunk,
                                 # Disable pipeline processes that won't be used
                                 disable=['toke2vec', 'tagger', 'attribute_ruler', 'lemmatizer', 'ner', 'entity_linker',
                                          'entity_ruler', 'textcat'],
                                 n_process=4
                                 ))

            for doc in docs:
                words = []
                sentences = doc.sents
                for sentence in sentences:
                    words += [str(token) for token in sentence]

                diff_words_set = set()

                for word in words:
                    syllable_count = syllables_count(word)
                    if word not in nlp.Defaults.stop_words and syllable_count >= 2:
                        diff_words_set.add(word)

                sentences_out.append(len(list(doc.sents)))
                diff_words_out.append(len(diff_words_set))

        return [sentences_out, diff_words_out]
```

From the features created using the above functions, we can calculate readability scores: 
- [Flesch Reading Ease (RE) score](https://readabilityformulas.com/flesch-reading-ease-readability-formula.php) =- 206.835 - (1.015 x `ASL`) - (84.6 x `ASW`) , where `ASL` = average sentence length and `ASW` = average syllables per word
- [Gunning Fog Readability Index](https://readabilityformulas.com/gunning-fog-readability-formula.php) = 0.4 x (average sentence length + percentage of difficult words) 

View the [FIX THIS LINK](https://google.com)   
GitHub Repo: [https://github.com/cmoroney/stackoverflow-questions](https://github.com/cmoroney/stackoverflow-questions)

