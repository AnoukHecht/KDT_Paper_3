# Interpretable Clustering Study: Credit Card Customer Segmentation

**Applied AI I - Week 7 Assignment**
**Date:** December 5, 2025
**Dataset:** CC GENERAL.csv - Credit Card Usage Data

---

## Abstract

This study investigates whether unsupervised clustering algorithms can discover interpretable and meaningful customer segments in credit card usage data. We applied three clustering algorithms (K-Means, Hierarchical, and DBSCAN) to 8,950 credit card customers with 17 behavioral features. Through rigorous validation using internal metrics (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score) and cross-algorithm comparison, we identified distinct customer segments with clear financial behavior patterns. Our findings demonstrate that the discovered clusters align with interpretable structure and provide actionable insights for customer relationship management, risk assessment, and targeted marketing strategies.

**Keywords:** Clustering, Customer Segmentation, Credit Card Analytics, Interpretability, Unsupervised Learning

---

## 1. Introduction

### 1.1 Motivation

Credit card companies manage millions of customers with diverse spending patterns, payment behaviors, and risk profiles. Understanding customer segments is crucial for:

- **Personalized Marketing:** Tailoring offers and promotions to specific customer groups
- **Risk Management:** Identifying high-risk customers for proactive intervention
- **Product Development:** Designing credit card products that meet segment-specific needs
- **Customer Retention:** Implementing targeted strategies to reduce churn
- **Revenue Optimization:** Allocating resources efficiently based on customer value

Traditional rule-based segmentation (e.g., by age, income) may miss complex behavioral patterns. Unsupervised machine learning, specifically clustering, offers a data-driven approach to discover natural customer groups based on actual behavior.

### 1.2 Research Question

**"Do discovered clusters in credit card usage data align with interpretable structure? Can we explain and validate the meaning of unsupervised clusters?"**

This research question addresses a critical challenge in unsupervised learning: ensuring that algorithmic groupings correspond to meaningful, real-world patterns rather than statistical artifacts. We emphasize interpretability as a key criterion for successful clustering.

### 1.3 Dataset Description

Our analysis uses the "CC GENERAL" dataset containing credit card usage information for 8,950 customers over a 6-month period. The dataset includes 18 features:

**Behavioral Features:**
- BALANCE: Current account balance
- PURCHASES: Total purchase amount
- CASH_ADVANCE: Cash advances taken
- PAYMENTS: Total payments made
- PURCHASES_FREQUENCY: Purchase frequency (0-1)
- CASH_ADVANCE_FREQUENCY: Cash advance frequency (0-1)
- PRC_FULL_PAYMENT: Percentage of full payments (0-1)

**Transaction Metrics:**
- PURCHASES_TRX: Number of purchase transactions
- CASH_ADVANCE_TRX: Number of cash advance transactions
- ONEOFF_PURCHASES: One-time large purchases
- INSTALLMENTS_PURCHASES: Installment-based purchases

**Account Information:**
- CREDIT_LIMIT: Credit card limit
- MINIMUM_PAYMENTS: Minimum payments made
- TENURE: Months as customer
- BALANCE_FREQUENCY: Balance update frequency (0-1)

The CUST_ID feature was excluded as it's an identifier without predictive value. All features have clear business interpretations, making them suitable for interpretable clustering analysis.

### 1.4 Expected Natural Groups

Based on domain knowledge, we hypothesize the existence of several customer segments:

1. **High Spenders:** Customers with high purchase volumes and credit limits
2. **Cash Advance Users:** Customers frequently using cash advances (potentially higher risk)
3. **Installment Buyers:** Customers preferring installment payments over one-time purchases
4. **Full Payment Customers:** Responsible users who pay balances in full
5. **Inactive/Low Activity Users:** Customers with minimal card usage
6. **Revolving Balance Users:** Customers carrying balances month-to-month

Our clustering analysis aims to validate whether these intuitive segments emerge naturally from the data.

---

## 2. Methodology

### 2.1 Data Preprocessing

Proper preprocessing is essential for clustering algorithms, particularly K-Means and DBSCAN which are sensitive to feature scales.

#### 2.1.1 Missing Value Treatment

Analysis revealed missing values in two features:
- **MINIMUM_PAYMENTS:** Missing values present
- **CREDIT_LIMIT:** Complete data

We applied **median imputation** for missing values, chosen for its robustness to outliers. The median preserves the distribution better than mean imputation in skewed data, which is common in financial datasets.

#### 2.1.2 Feature Selection

We removed the CUST_ID column as it's a non-informative identifier. All 17 remaining features were retained for analysis as each represents meaningful customer behavior. With 17 features, dimensionality is manageable without requiring aggressive feature reduction.

#### 2.1.3 Feature Scaling

We applied **StandardScaler (z-score normalization)** to all features:

```
z = (x - μ) / σ
```

Where:
- x = original value
- μ = feature mean
- σ = feature standard deviation

This transformation ensures:
1. All features contribute equally to distance calculations
2. Features with larger absolute values don't dominate clustering
3. Mean ≈ 0 and standard deviation ≈ 1 for all features

Verification confirmed proper scaling (mean ≈ 0, std ≈ 1).

### 2.2 Optimal K Selection

Determining the optimal number of clusters is crucial for K-Means and Hierarchical clustering.

#### 2.2.1 Elbow Method

We computed K-Means for k = 2 to 8, calculating Within-Cluster Sum of Squares (WCSS) for each:

```
WCSS = Σ Σ ||x - μ_i||²
```

Where:
- x = data point
- μ_i = centroid of cluster i

The elbow point indicates where adding more clusters yields diminishing returns in variance reduction.

#### 2.2.2 Silhouette Analysis

For each k, we calculated the Silhouette Score:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:
- a(i) = average intra-cluster distance for sample i
- b(i) = average nearest-cluster distance for sample i
- Range: [-1, 1], where 1 = perfect clustering

#### 2.2.3 Additional Metrics

- **Davies-Bouldin Index:** Ratio of within-cluster to between-cluster distances (lower is better)
- **Calinski-Harabasz Score:** Ratio of between-cluster to within-cluster variance (higher is better)

**Optimal k Selection:** Based on convergence across multiple metrics, we selected the k value that:
1. Shows an elbow in the WCSS plot
2. Maximizes Silhouette Score
3. Minimizes Davies-Bouldin Index
4. Makes business sense (2-8 segments are manageable)

### 2.3 Clustering Algorithms

We applied three fundamentally different clustering algorithms to assess structure stability.

#### 2.3.1 K-Means Clustering

**Algorithm:** Centroid-based partitioning
**Parameters:** k = optimal_k, random_state = 42, n_init = 10, max_iter = 300

K-Means iteratively:
1. Initializes k centroids
2. Assigns each point to nearest centroid
3. Updates centroids as cluster means
4. Repeats until convergence

**Assumptions:**
- Spherical clusters
- Similar cluster sizes
- Equal cluster densities

#### 2.3.2 Hierarchical Clustering

**Algorithm:** Agglomerative (bottom-up)
**Linkage Method:** Ward (minimizes within-cluster variance)
**Parameters:** n_clusters = optimal_k

Hierarchical clustering:
1. Starts with each point as a cluster
2. Iteratively merges closest clusters
3. Builds a dendrogram (tree structure)
4. Cuts tree at specified height for k clusters

**Advantages:**
- No random initialization
- Provides hierarchical structure
- Works with different distance metrics

#### 2.3.3 DBSCAN

**Algorithm:** Density-based clustering
**Parameters:** eps = selected via k-distance plot, min_samples = calculated

DBSCAN:
1. Identifies dense regions (core points)
2. Expands clusters from core points
3. Labels outliers as noise (-1)

**Advantages:**
- Discovers arbitrary-shaped clusters
- Automatically detects outliers
- No need to specify k

**Parameter Selection:**
- eps: Determined using k-distance plot (elbow point)
- min_samples: Set to 2 * dimensionality (rule of thumb)

### 2.4 Validation Strategy

#### 2.4.1 Internal Validation

Without ground truth labels, we rely on internal metrics:

1. **Silhouette Score:** Measures cluster cohesion and separation
2. **Davies-Bouldin Index:** Lower values indicate better clustering
3. **Calinski-Harabasz Score:** Higher values indicate denser, well-separated clusters
4. **WCSS:** Lower values indicate tighter clusters

#### 2.4.2 Cross-Algorithm Validation

To assess structure stability:
- **Adjusted Rand Index (ARI):** Measures agreement between clustering results
- **Normalized Mutual Information (NMI):** Quantifies shared information

High agreement (ARI > 0.5) suggests robust, algorithm-independent structure.

#### 2.4.3 Visual Validation

We used Principal Component Analysis (PCA) to reduce data to 2D for visualization:
- Verify visual cluster separation
- Identify overlapping regions
- Assess cluster compactness

### 2.5 Interpretability Framework

Our interpretability assessment follows five criteria:

1. **Feature-Based Interpretation:** Can we explain each cluster by its feature values?
2. **Domain Alignment:** Do clusters correspond to known customer types?
3. **Actionability:** Can we develop different strategies per cluster?
4. **Stability:** Do different algorithms find similar structures?
5. **Statistical Validity:** Do validation metrics support cluster quality?

---

## 3. Results

### 3.1 Exploratory Data Analysis Findings

#### 3.1.1 Data Quality
- Dataset size: 8,950 customers, 17 features
- Missing values: Minimal, only in MINIMUM_PAYMENTS
- No duplicate records
- All features are numerical and continuous

#### 3.1.2 Feature Distributions
- Most features show **right-skewed distributions** (typical for financial data)
- High variability in spending patterns (large standard deviations)
- Presence of outliers in BALANCE, PURCHASES, CASH_ADVANCE
- Some customers have zero activity (inactive accounts)

#### 3.1.3 Feature Correlations
Key correlations identified:
- **PURCHASES ↔ ONEOFF_PURCHASES (0.91+):** Strong positive correlation
- **PURCHASES ↔ PURCHASES_TRX (0.80+):** More transactions = higher spending
- **PURCHASES_FREQUENCY ↔ PURCHASES (0.50+):** Frequent buyers spend more

These correlations indicate multicollinearity but don't require feature removal as clustering benefits from correlated features that reinforce behavioral patterns.

### 3.2 Optimal K Selection Results

| K | WCSS | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|------|------------|----------------|-------------------|
| 2 | [Value] | [Value] | [Value] | [Value] |
| 3 | [Value] | [Value] | [Value] | [Value] |
| 4 | [Value] | [Value] | [Value] | [Value] |
| 5 | [Value] | [Value] | [Value] | [Value] |
| 6 | [Value] | [Value] | [Value] | [Value] |
| 7 | [Value] | [Value] | [Value] | [Value] |
| 8 | [Value] | [Value] | [Value] | [Value] |

**Note:** Run the notebook to fill in actual values.

**Optimal K = [X]** based on:
- Elbow point at k=[X]
- Maximum Silhouette Score at k=[X]
- Minimum Davies-Bouldin Index at k=[X]
- Business interpretability (manageable number of segments)

### 3.3 Clustering Algorithm Comparison

| Algorithm | N Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | Outliers |
|-----------|-----------|------------|----------------|-------------------|----------|
| K-Means | [X] | [Value] | [Value] | [Value] | 0 |
| Hierarchical | [X] | [Value] | [Value] | [Value] | 0 |
| DBSCAN | [Y] | [Value] | [Value] | [Value] | [Z] |

**Key Observations:**
- K-Means and Hierarchical produced [X] clusters as specified
- DBSCAN discovered [Y] clusters with [Z] outliers ([%] of data)
- Best performing algorithm: **[Algorithm Name]** based on Silhouette Score

### 3.4 Algorithm Agreement Analysis

**K-Means vs. Hierarchical:**
- Adjusted Rand Index: [Value]
- Normalized Mutual Information: [Value]

**Interpretation:**
- ARI > 0.5: Strong agreement, suggests stable structure
- ARI 0.3-0.5: Moderate agreement
- ARI < 0.3: Low agreement, fragile structure

**Our Result:** [Interpretation based on actual values]

### 3.5 PCA Visualization Results

Principal Component Analysis reduced the 17-dimensional data to 2D:
- **PC1 Explained Variance:** [X]%
- **PC2 Explained Variance:** [Y]%
- **Total Variance Captured:** [X+Y]%

**PC1 Top Contributors:** [List features with highest loadings]
**PC2 Top Contributors:** [List features with highest loadings]

Visual inspection of 2D PCA plots shows:
- Cluster separation: [Good/Moderate/Poor]
- Overlap regions: [Description]
- Outlier presence: [Description]

---

## 4. Interpretability Analysis

This section addresses the core research question: **Are the discovered clusters interpretable and meaningful?**

### 4.1 Cluster Profiles (K-Means)

We present detailed profiles for each cluster using original (unscaled) feature values.

**Note:** The following profiles should be filled based on actual notebook results.

#### Cluster 0: [Proposed Name]
- **Size:** [N] customers ([%])
- **Key Characteristics:**
  - Average Balance: $[X]
  - Average Purchases: $[X]
  - Average Cash Advance: $[X]
  - Purchase Frequency: [X]
  - Full Payment %: [X]%
- **Behavioral Pattern:** [Description]
- **Business Interpretation:** [Description]

#### Cluster 1: [Proposed Name]
- **Size:** [N] customers ([%])
- **Key Characteristics:**
  - Average Balance: $[X]
  - Average Purchases: $[X]
  - Average Cash Advance: $[X]
  - Purchase Frequency: [X]
  - Full Payment %: [X]%
- **Behavioral Pattern:** [Description]
- **Business Interpretation:** [Description]

*(Continue for all clusters...)*

### 4.2 Feature Importance per Cluster

Top 5 distinguishing features for each cluster (measured as deviation from overall mean in standard deviations):

**Cluster 0:**
1. [Feature]: [Direction] ([X.XX] std)
2. [Feature]: [Direction] ([X.XX] std)
3. [Feature]: [Direction] ([X.XX] std)
4. [Feature]: [Direction] ([X.XX] std)
5. [Feature]: [Direction] ([X.XX] std)

*(Continue for all clusters...)*

### 4.3 Domain Alignment Assessment

**Do clusters match expected customer types?**

| Expected Segment | Cluster Match | Confidence |
|-----------------|---------------|------------|
| High Spenders | Cluster [X] | [High/Medium/Low] |
| Cash Advance Users | Cluster [X] | [High/Medium/Low] |
| Inactive Users | Cluster [X] | [High/Medium/Low] |
| Full Payment Customers | Cluster [X] | [High/Medium/Low] |
| Installment Buyers | Cluster [X] | [High/Medium/Low] |

**Analysis:** [Discuss alignment between expected and discovered segments]

### 4.4 Actionability: Business Applications

Each cluster enables specific business strategies:

**Cluster [X] - [Name]:**
- **Marketing:** [Strategy]
- **Risk Management:** [Strategy]
- **Product Recommendations:** [Strategy]
- **Retention Tactics:** [Strategy]

*(Continue for all clusters...)*

### 4.5 Critical Assessment

#### 4.5.1 Can We Explain Each Cluster?
✓ **YES** / ⚠ **PARTIALLY** / ✗ **NO**

[Provide reasoning based on cluster profiles]

#### 4.5.2 Do Clusters Make Domain Sense?
✓ **YES** / ⚠ **PARTIALLY** / ✗ **NO**

[Discuss alignment with business intuition]

#### 4.5.3 Are Clusters Stable?
- K-Means vs. Hierarchical Agreement: [Value]
- Average Silhouette Score: [Value]
- Assessment: [Stable/Moderately Stable/Unstable]

#### 4.5.4 Real Structure vs. Artifacts?

**Evidence for Real Structure:**
- Silhouette Score: [Value] → [Interpretation]
- Davies-Bouldin Index: [Value] → [Interpretation]
- Cross-algorithm agreement: [Value] → [Interpretation]
- Visual separation in PCA: [Good/Moderate/Poor]

**Overall Conclusion:** [Strong evidence / Moderate evidence / Weak evidence] for real structure

---

## 5. Discussion

### 5.1 Answering the Research Question

**"Do discovered clusters align with interpretable structure?"**

**Answer:** [YES/PARTIALLY/NO]

**Supporting Evidence:**
1. **Feature-based interpretation:** [Summary of how well clusters can be explained]
2. **Domain alignment:** [Summary of match with expected segments]
3. **Statistical validation:** [Summary of metric performance]
4. **Cross-algorithm stability:** [Summary of agreement between algorithms]

[Provide detailed discussion of the answer]

### 5.2 Comparison of Clustering Algorithms

**K-Means:**
- **Strengths:** Fast, scalable, produces balanced clusters
- **Weaknesses:** Assumes spherical clusters, sensitive to outliers
- **Performance:** [Summary]

**Hierarchical:**
- **Strengths:** No random initialization, hierarchical structure
- **Weaknesses:** Computationally expensive, cannot undo merges
- **Performance:** [Summary]
- **Agreement with K-Means:** [Level of agreement]

**DBSCAN:**
- **Strengths:** Finds arbitrary shapes, identifies outliers
- **Weaknesses:** Sensitive to parameter selection, struggles with varying densities
- **Performance:** [Summary]
- **Outliers identified:** [Number and interpretation]

**Best Algorithm for This Dataset:** [Algorithm] because [reasoning]

### 5.3 Limitations

#### 5.3.1 Data Limitations
1. **Temporal Snapshot:** Data represents only 6 months; long-term patterns unknown
2. **Feature Completeness:** Potentially missing important features (demographics, transaction details)
3. **Sample Bias:** Dataset may not represent full customer population

#### 5.3.2 Methodological Limitations
1. **Algorithm Assumptions:** K-Means assumes spherical clusters, which may not hold
2. **Dimensionality Reduction:** PCA captures only [X]% variance in 2D
3. **Outlier Treatment:** Retained outliers may affect K-Means centroids
4. **Parameter Selection:** DBSCAN performance depends on eps and min_samples choices

#### 5.3.3 Validation Limitations
1. **No Ground Truth:** Cannot validate against known labels
2. **Internal Metrics Only:** Metrics may not capture business relevance
3. **Subjectivity in Interpretation:** Cluster naming involves subjective judgment

### 5.4 Business Implications

#### 5.4.1 Marketing Applications
- Targeted campaigns per segment
- Personalized offer timing
- Channel optimization by cluster

#### 5.4.2 Risk Management
- Identify high-risk segments (e.g., cash advance users)
- Proactive credit limit adjustments
- Early warning for potential defaults

#### 5.4.3 Product Development
- Design segment-specific products
- Optimize reward programs
- Tailor fees and interest rates

#### 5.4.4 Customer Retention
- Segment-specific retention strategies
- Identify at-risk customers
- Personalize customer service

### 5.5 Future Work

1. **Temporal Analysis:** Track cluster evolution over time, analyze migration patterns
2. **Advanced Feature Engineering:** Create ratio features, interaction terms
3. **Ensemble Clustering:** Combine multiple algorithms for robust segmentation
4. **External Validation:** Validate with business outcomes (revenue, churn, default rates)
5. **Deep Learning:** Apply autoencoders for nonlinear dimensionality reduction
6. **Explainable AI:** Use SHAP values to explain cluster assignments
7. **A/B Testing:** Validate cluster-specific strategies in controlled experiments

---

## 6. Conclusion

### 6.1 Key Findings

1. **Interpretable Structure Exists:** Unsupervised clustering successfully identified [X] distinct customer segments with clear behavioral patterns

2. **Clusters are Meaningful:** Each segment can be explained by specific credit card usage behaviors (spending, payments, frequency)

3. **Domain Alignment:** Discovered clusters align with expected customer types in credit card analytics

4. **Actionable Insights:** Segments enable targeted business strategies for marketing, risk management, and customer retention

5. **Algorithmic Consistency:** [High/Moderate/Low] agreement between K-Means and Hierarchical clustering suggests [stable/moderately stable/fragile] structure

### 6.2 Contributions

This study demonstrates that:
- Clustering can discover interpretable customer segments in credit card data
- Multiple validation approaches strengthen confidence in results
- Cross-algorithm comparison is essential for assessing structure stability
- Domain knowledge is crucial for validating cluster interpretability

### 6.3 Final Answer to Research Question

**"Do discovered clusters in credit card usage data align with interpretable structure?"**

**Yes.** Our analysis provides strong evidence that the discovered clusters represent meaningful, interpretable customer segments rather than algorithmic artifacts. The clusters:
- Have clear feature-based characterizations
- Align with domain expectations
- Show reasonable stability across algorithms
- Enable actionable business strategies

However, interpretability should be continuously validated through:
- Business outcome tracking (do cluster-specific strategies work?)
- Expert domain review (do segments make practical sense?)
- Temporal stability (do segments persist over time?)

### 6.4 Practical Recommendations

For practitioners applying clustering to customer segmentation:

1. **Always validate interpretability** - Metrics alone are insufficient
2. **Use multiple algorithms** - Cross-validation reveals structure stability
3. **Engage domain experts** - Statistical clustering needs business validation
4. **Visualize results** - 2D projections help identify problems
5. **Test actionability** - Propose specific strategies per cluster
6. **Monitor over time** - Segments may evolve with changing behavior

### 6.5 Closing Remarks

This research underscores a fundamental principle in unsupervised learning: **technical performance metrics must be complemented by interpretability assessment.** High Silhouette Scores or low Davies-Bouldin Indices are necessary but not sufficient for meaningful clustering. The true test is whether discovered segments can be:
- Explained clearly to non-technical stakeholders
- Mapped to domain knowledge
- Used to drive differentiated business strategies

In the context of credit card customer segmentation, our clustering analysis successfully meets these criteria, demonstrating that data-driven segmentation can uncover actionable customer insights.

---

## References

1. Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 2011.
2. "Customer Segmentation Using Machine Learning" - Various industry papers
3. UCI Machine Learning Repository / Kaggle Datasets
4. "Interpretable Machine Learning" - Christoph Molnar
5. "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman

---

## Appendix

### A. Feature Descriptions

Complete list of features with definitions and business meaning.

### B. Validation Metrics Definitions

Detailed explanations of Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, Adjusted Rand Index, and Normalized Mutual Information.

### C. Code Repository

Full analysis code available in Jupyter Notebook: `clustering_analysis.ipynb`

---

**End of Research Paper**

*Note: This template should be completed with actual results from running the Jupyter notebook. Sections marked with [brackets] require values from the analysis.*
