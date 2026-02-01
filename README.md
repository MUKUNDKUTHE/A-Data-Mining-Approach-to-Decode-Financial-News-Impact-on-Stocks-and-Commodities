Project Title: From Headlines to Markets: A Data Mining Approach to Decode Financial
News Impact on Stocks and Commodities

Developed by:
Mukund Kuthe PRN: 23070521082 Section: B SEM: V
Ravi Kumar Kushwaha PRN: 23070521116 Section: B SEM: V

Project Overview
This project integrates structured, semi-structured, and unstructured financial data to
analyze how financial news sentiments influence market performance across equity,
derivatives, and commodities. It combines data warehousing, machine learning, and data
mining techniques to uncover meaningful patterns between financial sentiment and trading
behavior.

Project Components
• Data_Warehouse_and_Data_Mining_Project_Report.pdf: Detailed project documentation
covering background, methodology, ETL workflow, data warehouse design, data mining
algorithms, results, and future scope.
• Data Science PPT.pptx: PowerPoint presentation summarizing project workflow,
datasets, data mining visuals, and Power BI dashboard.
• classification.py: Implements Random Forest Classification to predict financial news
sentiment based on trading metrics.
• classification1.py: Implements XGBoost Classifier to predict Market Performance and
show feature importance.
• k_means_cluster.py: Performs K-Means Clustering on trading activity data to group
markets by performance.
• rule_mining.py: Applies Apriori Association Rule Mining to discover associations
between instrument type, segment, and commodity.

Technologies & Tools Used
Python (Pandas, NumPy, scikit-learn, XGBoost, mlxtend, Matplotlib, Seaborn)
SQL Server, SSIS, SSAS, Power BI

Algorithms Implemented
Random Forest - Sentiment Classification (Accuracy: 100%)
XGBoost - Market Performance Prediction (Key Feature: st_delivered_value_crores)
K-Means - Market Segmentation (Silhouette Score: 0.434)
Apriori - Association Rule Mining (Max Lift: 7.11)

Datasets Used
• Financial News (Kaggle) – Unstructured (CSV)
• Capital Market (NSE) – Semi-structured (JSON)
• Futures & Options (NSE) – Semi-structured (JSON)
• Commodity Market (MCX) – Structured (Excel)

How to Run the Python Scripts
1. Place 'data.csv'(Combine data) in the same directory as scripts.
2. Install dependencies using:
 pip install pandas numpy scikit-learn xgboost mlxtend matplotlib seaborn
3. Run:
 python classification.py
 python classification1.py
 python k_means_cluster.py
 python rule_mining.py

Key Outcomes
• Automated ETL pipeline for financial data
• Unified SQL Server data warehouse for sentiment and market metrics
• Machine learning models establish relationships between news sentiment and market
behavior
• Power BI dashboards for interactive financial analysis
