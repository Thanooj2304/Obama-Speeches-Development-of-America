# Obama-Speeches-Development-of-America
This repository uses text and web analytics such as sentiment analysis and much more to see how much importance is given by obama for the development of America
1. Introduction & Context
This report presents a comprehensive Text & Web Analytics study of Obama's Development of America Speeches dataset   a rich corpus spanning from 2009 to 2016. The dataset, sourced entirely from obamawhitehouse.archives.gov, contains over 13,400 presidential communications including Statements, Speeches, Press Briefings, Proclamations, and Executive Orders.
The primary objective of this project is to apply the full spectrum of Natural Language Processing (NLP) and Text Analytics techniques taught under the Text and Web Analytics subject to derive meaningful insights from real-world political discourse. By examining how President Obama communicated development priorities   from economic recovery to infrastructure, healthcare, education, and clean energy   this report demonstrates the power of computational text analysis as a business intelligence tool.
Key analytical goals include understanding the structure and distribution of the dataset, preprocessing raw text for downstream modelling, applying representation techniques such as Bag-of-Words, TF-IDF, and Word2Vec, performing sentiment and aspect-based analysis, discovering latent topics through LDA and LSA, clustering documents, and identifying named entities.

2. Dataset Overview   Structured Data Analysis
The dataset encompasses approximately 13,405 documents scraped from the official Obama White House archives. As illustrated below, the corpus is dominated by Statements and Releases (~12,000 documents), followed by Speeches and Remarks (~4,200), Press Briefings, Proclamations, Presidential Memoranda, Weekly Addresses, and Executive Orders. This distribution reflects the administrative reality that formal releases outnumber public speeches significantly.
 
Figure 1: Document type distribution (left) and development speeches by year (right)
The temporal distribution of development-related speeches across years 2009–2016 shows peak activity in 2010 (1,862 speeches) and 2014 (1,783 speeches), with a steep drop in 2016 (68) attributable to the final months of the administration. This temporal pattern provides important contexts for all downstream analyses, as content and priority themes shifted across the two-term presidency.
Structured vs. Unstructured Data
This dataset exemplifies the distinction central to text analytics: structured data includes fields like document type, date, and URL, while unstructured data is the raw speech text. Business value is unlocked by applying NLP to the unstructured content and joining insights with the structured metadata.
3. Text Preprocessing
Raw text data requires systematic preprocessing before any meaningful analysis can be conducted. This section covers the four core preprocessing stages: tokenisation, stop word removal, stemming, and lemmatization each illustrated with figures drawn from the project code.
3.1 Tokenisation
Tokenisation is the foundational step of converting raw text into a sequence of discrete units (tokens). The analysis employed word-level tokenisation using NLTK. Figure 2 shows the top 30-word tokens from a sample speech before any filtering   function words like 'and', 'the', 'to', and 'in' dominate, illustrating why stop word removal is essential as a next step.
 
Figure 2: Top 30 word tokens before stop word removal function words dominate
3.2 Stop Word Removal
Stop words are high-frequency function words (articles, prepositions, conjunctions) that carry minimal semantic content. After applying NLTK's English stop word list, the most frequent terms shift dramatically to substantive words: 'argentina', 'trade', 'president', 'economic', 'investment', and 'united' emerge as the dominant terms in the sample speech about US–Argentina relations.
 
Figure 3: Token frequency comparison before (left) and after (right) stop word removal
3.3 Stemming vs. Lemmatisation
Both stemming and lemmatisation aim to normalise word forms to their base representation. Stemming (Porter algorithm) aggressively truncates suffixes   'investment' becomes 'invest', 'president' becomes 'presid', 'economic' becomes 'econom'. Lemmatisation uses linguistic rules to produce valid dictionary forms   'united' becomes 'unite', preserving readability. The vocabulary sizes are comparable (120 vs. 121), but lemmatisation produces more interpretable output, making it preferable for topic modelling and sentiment analysis.
 
Figure 4: Vocabulary comparison   original tokens, after stemming, and after lemmatisation
4. Feature Representation   BoW, TF-IDF & Word Embeddings
4.1 Bag of Words (BoW)
The Bag-of-Words model represents each document as a vector of word frequencies, ignoring grammar and word order. Applied to the full corpus of Development Speeches, 'president' is by far the most frequent term (188,014 occurrences), followed by 'make' (112,156), 'think' (97,438), and 'state' (93,049). These high frequencies reflect the formal register of presidential rhetoric and the recurring themes of national direction and unity.
 
Figure 5: Top 20 terms by total frequency across all development speeches (Bag-of-Words)

4.2 TF-IDF (Term Frequency–Inverse Document Frequency)
TF-IDF improves on BoW by down-weighting terms that appear in many documents (common across the corpus) and up-weighting terms that are distinctive to specific documents. The heatmap below reveals the 'signature terms' for each speech   terms that are simultaneously frequent in that document but rare elsewhere. Notice how 'obama', 'security', 'carbon', and 'shutdown' are highly specific to particular speech types, providing a more nuanced fingerprint than raw frequencies.
 
Figure 6: TF-IDF heatmap showing signature terms per speech   lighter cells indicate higher TF-IDF scores
4.3 N-Gram Analysis
N-gram analysis extends the BoW model to capture phrase-level patterns. Unigrams confirm 'president' as the dominant single token. Bigrams reveal institutionally significant phrases: 'united states', 'president obama', 'white house', 'health care', 'middle class'. Trigrams show compound proper nouns: 'united states America', 'president barack obama', 'white house office', 'affordable care act'   precisely the policy terminology expected of development-themed presidential communications.
 
Figure 7: Top N-grams by mean TF-IDF   unigrams (left), bigrams (center), trigrams (right)
4.4 Word Embeddings   Word2Vec
Word2Vec trains a neural network to learn dense vector representations of words based on co-occurrence context. The t-SNE projection of the trained embeddings onto two dimensions reveals semantic clustering: 'economy' and 'growth' appear together in the upper region; 'healthcare' and 'education' cluster nearby; 'budget', 'tax', and 'invest' form a financial cluster; 'energy' and 'innovation' are proximal. These spatial relationships capture the semantic relationships that simpler BoW models cannot represent.
 
Figure 8: t-SNE projection of Word2Vec embeddings   semantically related policy terms cluster together
5. Text Classification
Text classification assigns documents to predefined categories based on their content. Three standard classifiers were evaluated using 5-fold cross-validation on the TF-IDF representation of the corpus.
Naive Bayes achieved an accuracy of 51.7%, performing at near-chance level   expected, as its strong independence assumption is violated by political text where word co-occurrences are structurally important. Both SVM (LinearSVC) and Logistic Regression achieved 64.6% and 64.8% accuracy respectively, demonstrating that linear classifiers with TF-IDF features provide a meaningful baseline for document-type classification from speech content alone.
 
Figure 9: Text classification accuracy comparison   Naive Bayes, SVM, and Logistic Regression (5-fold CV)
Business Application
Document classification has direct business applications in legal document routing, customer support ticket categorization, news article tagging, and email filtering. The SVM and Logistic Regression results here (~65%) represent a meaningful starting point that would improve with feature engineering or transformer-based models.
6. Sentiment Analysis
6.1 VADER Rule-Based Sentiment
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analyzer optimized for short, informal texts but also effective for formal political speech. Applied to the full corpus of development speeches, results are remarkably consistent: 92% of speeches are classified as 'Optimistic' (compound score > +0.15), 7% as 'Urgent' (< -0.05), and only 1% as Neutral.
 
Figure 10: VADER sentiment analysis   distribution (left), compound scores per speech (center), average sentiment arc by year (right)
The average compound sentiment arc by year shows consistent positivity (~0.8) throughout 2009–2015, with a slight dip in 2016 (to ~0.7), which coincides with politically turbulent campaign-year discourse. This sustained positivity is consistent with the aspirational, forward-looking rhetoric characteristic of presidential development addresses.
6.2 Aspect-Based Sentiment Analysis
Aspect-based sentiment analysis goes beyond document-level sentiment to measure sentiment toward specific policy domains within each speech. Six domains were tracked: Economy, Jobs, Education, Healthcare, Energy, and Infrastructure. The heatmap reveals nuanced patterns: the FACT SHEET ON SYRIA shows negative sentiment on Economy (red), while the Affordable Care Act and healthcare memoranda show strongly positive Healthcare sentiment (dark green). Energy sentiment is predominantly positive across addresses related to clean energy initiatives.
 
Figure 11: Aspect-based sentiment heatmap   policy domains across individual speeches (green = positive, red = negative)
7. Topic Modelling
7.1 LDA   Latent Dirichlet Allocation
LDA is a generative probabilistic model that discovers latent topics in a document corpus. Applied to the development speeches corpus, LDA identified five coherent topics: Economic Recovery & Jobs (4,399 speeches   the dominant topic), Trade, Innovation & Competitiveness (3,637), Infrastructure & Clean Energy (2,237), Education & Workforce (1,702), and Healthcare & Social Safety (1,430).


 
Figure 12: LDA topic modelling   topic distribution (left) and top word probabilities per topic (right)
The word probability heatmap (right) shows that 'make', 'people', 'work', 'want', and 'think' are the highest-probability words within the Economic Recovery & Jobs topic, reflecting the conversational, populist register of Obama's economic speeches. The Healthcare & Social Safety topic is characterized by slightly higher probabilities (0.018) for its top terms, suggesting greater lexical specificity in health-related communications.
7.2 LSA   Latent Semantic Analysis
LSA applies Singular Value Decomposition (SVD) to the TF-IDF matrix to discover latent semantic dimensions. The document space projection onto Component 1 vs. Component 2 reveals a structured landscape: Executive Orders (yellow) and Weekly Addresses (green) cluster distinctly from Press Briefings (orange) and formal Statements (magenta). This confirms that document type carries significant latent semantic signal   an important finding for classification and retrieval applications.
 
Figure 13: LSA document space   Component 1 vs. Component 2, colored by document type
7.3 K-Means Document Clustering
K-Means partitions documents into k clusters by minimizing within-cluster sum of squares. The elbow method identified K=4 as optimal   the inflection points where marginal inertia reduction diminishes. The resulting cluster sizes are unequal: Cluster 2 contains 6,697 documents (the broad general-address cluster), Cluster 3 has 3,308, Cluster 1 has 2,299, and Cluster 0 has 1,101. This imbalance is typical of real-world political corpora where a large 'catch-all' cluster of general speeches co-exists with more specialized sub-clusters.

 
Figure 14: K-Means clustering   elbow method (left) and final cluster sizes at K=4 (right)
7.4 Hierarchical Clustering
Hierarchical clustering builds a tree (dendrogram) of document similarity using Ward linkage, which merges clusters to minimize within-cluster variance. The dendrogram reveals three major super-clusters: a large red cluster of Weekly Addresses and Presidential Remarks, a green cluster of early 2009 Press Briefings, and an orange cluster of FACT SHEETs and Presidential Memoranda. This structure validates the LDA and K-Means results, showing that document type and policy domain jointly determines textual similarity.
 
Figure 15: Hierarchical clustering dendrogram   Ward linkage over sample of Obama development speeches
8. Named Entity Recognition (NER)
Named Entity Recognition identifies and classifies named entities in text into categories such as persons, locations, organizations, monetary values, and geopolitical entities. Using spaCy's ML-based NER pipeline, six entity types were extracted and visualized across the development speeches corpus.
 
Figure 16: NER analysis   six entity categories across Obama development speeches
Geopolitical entities (GPE): 'The United States' dominates with 178 mentions, followed by 'America' (62), 'Argentina' (49), 'China' (39), and 'U.S.' (37)   reflecting the international trade and bilateral relations context of many speeches. Organizations: 'Congress' leads with 98 mentions, confirming the legislative-executive dynamic as central to development discourse. Persons: 'Barack Obama' and 'Barack' (24 each) and 'Obama' (23) represent the self-referential rhetoric of presidential addresses. Political groups (NORP): 'American(s)' (88+73) and 'Republicans' (59) highlight the bipartisan framing of development policy. Monetary values include references to specific investments like '$23 billion' and 'A Trillion Dollars', quantifying the financial scale of development commitments.
9. Word Clouds   Visual Frequency Analysis
Word clouds provide an intuitive visual summary of term frequency, with font size proportional to occurrence count. Six-word clouds were generated: one for the full corpus and one per LDA topic. The all-corpus cloud confirms 'president', 'make', 'people', 'state', 'work', and 'think' as the lexical core of development discourse.
 
Figure 17: Word clouds for all development speeches and each LDA topic   font size proportional to frequency
Topic-specific clouds reveal distinguishing vocabulary: the Economic Recovery cloud centers on 'go', 'make', and 'work'; Infrastructure & Clean Energy foregrounds 'energy', 'new', and 'support'; Education & Workforce emphasizes 'state', 'federal', and 'service'; Healthcare & Social Safety highlights 'university', 'director', and 'serve'; Trade, Innovation & Competitiveness features 'president', 'think', and 'say'. These visualisations provide accessible communication of topic modelling results for non-technical stakeholders.
10. Web Scraping   Data Collection
The entire corpus was assembled via web scraping of obamawhitehouse.archives.gov. The source domain chart confirms that all 13,400+ documents originate from a single canonical source, validating the dataset's coherence and provenance. The scraping pipeline employed BeautifulSoup for HTML parsing, with respectful rate-limiting and compliance with the site's robots.txt directives.
 
Figure 18: Source domains in the dataset   all documents from obamawhitehouse.archives.gov
Web Scraping Ethics
Ethical web scraping requires: (1) consulting robots.txt before crawling, (2) implementing rate limiting to avoid server overload, (3) identifying the scraper with a User-Agent string, (4) using only publicly accessible data, and (5) respecting copyright and terms of service. All these principles were followed by collecting this dataset.
11. Interpretations & Key Insights
Synthesizing across all analytical techniques, several important insights emerge from this study:
Thematic dominance of economic discourse: LDA, BoW, and N-gram analysis consistently identify Economic Recovery & Jobs as the largest and most linguistically prominent topic, reflecting the post-2008 recession context of the Obama presidency.
Sustained optimism: VADER sentiment analysis reveals that 92% of development speeches carry a positive compound score, confirming that presidential communications are systematically aspirational   a finding with implications for political communication research and public sentiment modelling.
Policy-specific language sharpens over time: TF-IDF and aspect-based sentiment analysis reveal that specific policy domains (healthcare, energy, infrastructure) generated increasingly specialized vocabulary, particularly after the Affordable Care Act passage in 2010.
Document type as a strong latent signal: Both LSA and hierarchical clustering confirm that document type (Weekly Address, Press Briefing, FACT SHEET, Executive Order) produces coherent textual clusters, suggesting that genre conventions shape linguistic choices as much as topical content.
Word2Vec captures policy semantic space: The t-SNE embedding map reveals that Obama's development vocabulary organizes around coherent policy clusters (economic, social, governance) a structure invisible to bag-of-words models but potentially highly valuable for semantic search and recommendation applications.
12. Conclusion
This project has demonstrated the comprehensive application of Text & Web Analytics techniques to a real-world corpus of presidential development speeches. Beginning with raw HTML content scraped from the official Obama White House archives, the pipeline progressed through systematic preprocessing, feature representation, classification, sentiment analysis, topic modelling, clustering, and named entity recognition.
The results confirm that computational text analysis can surface meaningful patterns in large political corpora   patterns not readily apparent through manual reading. The dominance of economic recovery themes, the sustained positive sentiment, the five coherent policy topics identified by LDA, and the semantic structure revealed by Word2Vec embeddings together constitute a richly textured portrait of how the Obama administration communicated its development vision to the American public.
From a business perspective, the methods demonstrated here   TF-IDF for document fingerprinting, SVM for classification, VADER for sentiment monitoring, LDA for topic discovery, and NER for entity extraction   represent a mature and deployable toolkit applicable to customer feedback analysis, competitive intelligence, regulatory document monitoring, and media analysis across industries.
SDG Relevance
This project aligns with SDG 16 (Peace, Justice and Strong Institutions) by analyzing government communications transparency, SDG 8 (Decent Work and Economic Growth) by tracking economic development discourse, SDG 4 (Quality Education) through education policy analysis, and SDG 3 (Good Health and Well-Being) through healthcare sentiment tracking.
13. References
Obama White House Archives. (2009–2016). Presidential Speeches and Remarks. Retrieved from obamawhitehouse.archives.gov
Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly Media.
Blei, D.M., Ng, A.Y., & Jordan, M.I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993–1022.
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM-14.
Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. ICLR Workshop.
Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.
Richardson, L. (2007). Beautiful Soup Documentation. Crummy.com.
Spärck Jones, K. (1972). A Statistical Interpretation of Term Specificity and Its Application in Retrieval. Journal of Documentation, 28(1), 11–21.
