import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


## Functions to get results ---------------------------------------------------------------------------

def determine_top_k_right(all_candidates, best_match, top_k, only_perc=False):
    if len(all_candidates) != len(best_match):
        raise ValueError("Length mismatch between candidates and best_match.")

    top_k_candidates = [candidates[:top_k] for candidates in all_candidates]

    top_k_right = [best_match[i] in top_k_candidates[i] for i in range(len(top_k_candidates))]

    if only_perc:
        return str(round(sum(top_k_right) / len(top_k_candidates) * 100, ndigits=1)) + "%"
    else:
        return str(sum(top_k_right)) + "/" + str(len(top_k_candidates)) + " (" + str(round(sum(top_k_right) / len(top_k_candidates) * 100, ndigits=1)) + "%)"

def add_best_match(mapping_data, label_data, df_foodon_combined, name_column):
    # Only keep those that have a best_match or a propose status
    label_mask = (
        (pd.notna(label_data['best_match'])) |
        (label_data['label'].str.lower().str.contains('propose'))
    )

    label_matches = label_data[label_mask].reset_index()

    label_matches.loc[:,'best_match'] = [label_matches['randomized_candidates'][i] if (pd.notna(label_data['best_match'][i])) else label_matches['proposed_foodon'][i] for i in range(label_matches.shape[0])]
    
    food_items = label_matches[[name_column, 'best_match', 'label', 'proposed_foodon_label']]
    
    merged_data = pd.merge(food_items, mapping_data, how="left", left_on=name_column, right_on="orig_name")
    merged_data = pd.merge(merged_data, df_foodon_combined[['name', 'Preferred Label']], how="left", left_on = 'best_match', right_on='Preferred Label')

    true_labels = np.where(
        pd.isna(merged_data['proposed_foodon_label']),
        merged_data['best_match'],
        merged_data['name']
    )
    
    merged_data.loc[:,'pos_candidates'] = [merged_data.loc[i,'candidates'].index(true_labels[i]) if true_labels[i] in merged_data.loc[i, 'candidates'] else np.nan for i in range(merged_data.shape[0])]
    merged_data.loc[:,'pos_reranked_candidates'] = [merged_data.loc[i,'reranked_candidates'].index(true_labels[i]) if true_labels[i] in merged_data.loc[i, 'reranked_candidates'] else np.nan for i in range(merged_data.shape[0])]
    merged_data.loc[:,'pos_hybrid_candidates'] = [merged_data.loc[i,'hybrid_candidates'].index(true_labels[i]) if true_labels[i] in merged_data.loc[i, 'hybrid_candidates'] else np.nan for i in range(merged_data.shape[0])]
    
    # Strip labels
    merged_data.loc[:, 'label'] = merged_data['label'].str.strip()
    
    return merged_data

def determine_results(merged_data, labels = ['broader', 'exact', 'close match', 'related', 'narrow'], only_perc=False):
    merged_data = merged_data[merged_data['label'].isin(labels)]

    orig_candidates = list(merged_data['candidates'])
    reranked_candidates = list(merged_data['reranked_candidates'])
    hybrid_candidates = list(merged_data['hybrid_candidates'])

    best_match = list(merged_data['best_match'])
    
    return [determine_top_k_right(orig_candidates, best_match, k, only_perc) for k in [1,3,5]], \
           [determine_top_k_right(reranked_candidates, best_match, k, only_perc) for k in [1,3,5]], \
           [determine_top_k_right(hybrid_candidates, best_match, k, only_perc) for k in [1,3,5]]

def determine_results_traditional(merged_data, labels = ['broader', 'exact', 'close match', 'related', 'narrow'], only_perc=False):
    merged_data = merged_data[merged_data['final_label'].isin(labels)]

    orig_candidates = list(merged_data['candidates'])

    best_match = list(merged_data['best_match'])
    
    return [determine_top_k_right(orig_candidates, best_match, k, only_perc) for k in [1,3,5,15]]

def mean_index_of_candidate(candidates_list, similarities_list, candidate):
    # Find the similarity of the given candidate
    candidate_idx = candidates_list.index(candidate)

    similarity = similarities_list[candidate_idx]
    # Find all indices with the same similarity
    indices = [i for i, sim in enumerate(similarities_list) if sim == similarity]
    # Calculate mean index
    mean_idx = sum(indices) / len(indices)
    return mean_idx

def add_best_match_traditional(mapping_data, label_data, df_foodon_combined, name_column):
    merged_data = pd.merge(label_data, mapping_data, how="left", left_on=name_column, right_on="orig_name")

    # For the propose labels, add the 'name' column which is the name we used after preprocessing
    merged_data = pd.merge(merged_data, df_foodon_combined[['name', 'Preferred Label']], how="left", left_on = 'best_match', right_on='Preferred Label')

    # The best matches are in column 'best_match' when label is not 'propose', otherwise in column 'name'
    best_matches = np.where(
        merged_data['label'] != 'propose',
        merged_data['best_match'],
        merged_data['name']
    )

    # Get indices of candidates. If multiple candidates have the same similarity score, take the mean index
    pos_candidates = []
    for i in range(merged_data.shape[0]):
        candidate = best_matches[i]
        candidates_list = merged_data.loc[i, 'candidates']
        similarities_list = merged_data.loc[i, 'similarity']
        if pd.notna(candidate) and candidate in candidates_list:
            pos_candidates.append(mean_index_of_candidate(candidates_list, similarities_list, candidate))
        else:
            pos_candidates.append(np.nan)

    merged_data['pos_candidates'] = pos_candidates

    # Strip labels
    merged_data.loc[:, 'label'] = merged_data['label'].str.strip()

    return merged_data
    

## Classes for Similarity Processors ------------------------------------------------

class OpenAITextSimilarityProcessor:
    def __init__(self, df1, df2, df1_embedding_column, df2_embedding_column):
        self.df1 = df1
        self.df2 = df2
        self.df1_vec = np.array(df1[df1_embedding_column].tolist())
        if len(self.df1_vec.shape) == 3:
            self.df1_vec = self.df1_vec[:,0,:] # Remove second dimension
        self.df2_vec = np.array(df2[df2_embedding_column].tolist())
        if len(self.df2_vec.shape) == 3:
            self.df2_vec = self.df2_vec[:,0,:] # Remove second dimension
        self.similarity_matrix = None
        self.subset_similarity_matrix = None
        self.results_df = None

    def compute_cosine_similarity(self):
        """Compute cosine similarity matrix between df1 and df2 BoW/TF-IDF vectors."""
        self.similarity_matrix = cosine_similarity(self.df1_vec, self.df2_vec)

    def find_top_k(self, names, top_k=-1):
        """Find top 15 most similar foodon items (from df2) for each food item in df1"""
        if top_k == -1:
            top_k = self.df2.shape[0]
        
        # Get indices in df1 from given names
        df1_indices = [list(self.df1['name']).index(name) for name in names]

        # Get subset from similarity matrix to determine top k
        self.subset_similarity_matrix = self.similarity_matrix[df1_indices]

        top_indices = np.argsort(self.subset_similarity_matrix, axis=1)[:, -top_k:][:, ::-1]
        results = []
        for i, index in enumerate(df1_indices): # Index for original df1, i for subset_similarity_matrix and top_indices
            # Indices of (top / sorted) candidate items in FoodOn for food item i in the nevo/fdc/kap subset
            indices = top_indices[i]
            # Enumerate over indices of sorted candidate FoodOn items (per food item from nevo/fdc/kap)
            for idx in indices:
                results.append({
                    'df1_label': self.df1.iloc[index]['name'], # Food item from nevo/fdc/kap, 'index' because taken from original df
                    'df2_label': self.df2.iloc[idx]['name'], # Iteratively add next most similar candidate from FoodOn
                    'similarity': self.subset_similarity_matrix[i, idx] # subset_similarity_matrix is subset, so use 'i'
                })
        self.results_df = pd.DataFrame(results)

    def aggregate_results(self):
        if self.results_df is not None:
            self.results_df = self.results_df.groupby(
                ["df1_label"]
            )[["df2_label", "similarity"]].agg(list).reset_index(drop=False)
        else:
            raise ValueError("Results are not available. Please run find_top_k first.")

    def save_results(self, output_path, file_format='json'):
        if self.results_df is not None:
            if file_format == 'json':
                self.results_df.to_json(output_path, orient='records', lines=True)
            elif file_format == 'csv':
                self.results_df.to_csv(output_path, index=False)
            else:
                raise ValueError("Unsupported file format. Use 'json' or 'csv'.")
        else:
            raise ValueError("Results are not available. Please run find_top_similar first.")

    def get_results(self):
        return self.results_df
    
class TextSimilarityProcessor:
    def __init__(self, df1, df2, vectorizer_type):
        self.df1 = df1
        self.df2 = df2
        self.df1_clean = self.preprocess(df1)
        self.df2_clean = self.preprocess(df2)
        self.vectorizer_type = vectorizer_type
        self.vectorizer = None # chose 'bow' or 'tfidf'
        self.df1_vec = None
        self.df2_vec = None
        self.similarity_matrix = None
        self.results_df = None

    def preprocess(self, df_names):
        # To lowercase
        df_names = df_names.str.lower()
        # Remove brackets and other punctuation
        df_names = df_names.str.replace(r'[^\w\s]', '', regex=True)
        return df_names

    def fit_vectorizer(self):
        """Fit the chose vectorizer to the combined corpus from both datasets"""
        corpus = pd.concat([self.df1_clean, self.df2_clean]).astype(str).tolist()
        if self.vectorizer_type == 'bow':
            self.vectorizer = CountVectorizer()
        elif self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        else:
            raise ValueError("Unsupported vectorizer_type. Use 'bow' or 'tfidf'.")
        self.vectorizer.fit(corpus)

    def transform_to_vec(self):
        """Transform both columns to BoW vectors using the fitted vectorizer."""
        self.df1_vec = self.vectorizer.transform(self.df1_clean.astype(str))
        self.df2_vec = self.vectorizer.transform(self.df2_clean.astype(str))

    def compute_cosine_similarity(self):
        """Compute cosine similarity matrix between df1 and df2 BoW/TF-IDF vectors."""
        self.similarity_matrix = cosine_similarity(self.df1_vec, self.df2_vec)

    def find_top15(self, top_k=15):
        """Find top 15 most similar foodon items (from df2) for each food item in df1"""
        top_indices = np.argsort(self.similarity_matrix, axis=1)[:, -top_k:][:, ::-1]
        top_similarities = np.take_along_axis(self.similarity_matrix, top_indices, axis=1)
        results = []
        for i, indices in enumerate(top_indices):
            for rank, idx in enumerate(indices):
                results.append({
                    'df1_label': self.df1.iloc[i],
                    'df2_label': self.df2.iloc[idx],
                    'similarity': top_similarities[i, rank]
                })
        self.results_df = pd.DataFrame(results)

    def find_top_k(self, names, top_k=-1):
        """Find top 15 most similar foodon items (from df2) for each food item in df1"""
        if top_k == -1:
            top_k = self.df2.shape[0]
        
        # Get indices in df1 from given names
        df1_indices = [list(self.df1).index(name) for name in names]

        # Get subset from similarity matrix to determine top k
        self.subset_similarity_matrix = self.similarity_matrix[df1_indices]

        top_indices = np.argsort(self.subset_similarity_matrix, axis=1)[:, -top_k:][:, ::-1]
        results = []
        for i, index in enumerate(df1_indices): # Index for original df1, i for top_similarities and top_indices
            indices = top_indices[i]
            for idx in indices:
                results.append({
                    'df1_label': self.df1.iloc[index],
                    'df2_label': self.df2.iloc[idx],
                    'similarity': self.subset_similarity_matrix[i, idx]
                })
        self.results_df = pd.DataFrame(results)

    def aggregate_results(self):
        if self.results_df is not None:
            self.results_df = self.results_df.groupby(
                ["df1_label"]
            )[["df2_label", "similarity"]].agg(list).reset_index(drop=False)
        else:
            raise ValueError("Results are not available. Please run find_top15 first.")

    def save_results(self, output_path, file_format='json'):
        if self.results_df is not None:
            if file_format == 'json':
                self.results_df.to_json(output_path, orient='records', lines=True)
            elif file_format == 'csv':
                self.results_df.to_csv(output_path, index=False)
            else:
                raise ValueError("Unsupported file format. Use 'json' or 'csv'.")
        else:
            raise ValueError("Results are not available. Please run find_top_similar first.")

    def get_results(self):
        return self.results_df
    
class BERTTextSimilarityProcessor:
    def __init__(self, df1, df2, df1_embedding_column, df2_embedding_column):
        self.df1 = df1
        self.df2 = df2
        self.df1_vec = df1[df1_embedding_column].tolist()
        self.df2_vec = df2[df2_embedding_column].tolist()
        self.similarity_matrix = None
        self.results_df = None

    def compute_cosine_similarity(self):
        """Compute cosine similarity matrix between df1 and df2 BoW/TF-IDF vectors."""
        self.similarity_matrix = cosine_similarity(self.df1_vec, self.df2_vec)

    def find_top15(self, top_k=15):
        """Find top 15 most similar foodon items (from df2) for each food item in df1"""
        top_indices = np.argsort(self.similarity_matrix, axis=1)[:, -top_k:][:, ::-1]
        top_similarities = np.take_along_axis(self.similarity_matrix, top_indices, axis=1)
        results = []
        for i, indices in enumerate(top_indices):
            for rank, idx in enumerate(indices):
                results.append({
                    'df1_label': self.df1.iloc[i]['name'],
                    'df2_label': self.df2.iloc[idx]['name'],
                    'similarity': top_similarities[i, rank]
                })
        self.results_df = pd.DataFrame(results)

    def find_top_k(self, names, top_k=-1):
        """Find top 15 most similar foodon items (from df2) for each food item in df1"""
        if top_k == -1:
            top_k = self.df2.shape[0]
        
        # Get indices in df1 from given names
        df1_indices = [list(self.df1['name']).index(name) for name in names]

        # Get subset from similarity matrix to determine top k
        self.subset_similarity_matrix = self.similarity_matrix[df1_indices]

        top_indices = np.argsort(self.subset_similarity_matrix, axis=1)[:, -top_k:][:, ::-1]
        
        results = []
        for i, index in enumerate(df1_indices): # Index for original df1, i for top_similarities and top_indices
            indices = top_indices[i]
            for idx in indices:
                results.append({
                    'df1_label': self.df1.iloc[index]['name'],
                    'df2_label': self.df2.iloc[idx]['name'],
                    'similarity': self.subset_similarity_matrix[i, idx]
                })
        self.results_df = pd.DataFrame(results)

    def aggregate_results(self):
        if self.results_df is not None:
            self.results_df = self.results_df.groupby(
                ["df1_label"]
            )[["df2_label", "similarity"]].agg(list).reset_index(drop=False)
        else:
            raise ValueError("Results are not available. Please run find_top15 first.")

    def save_results(self, output_path, file_format='json'):
        if self.results_df is not None:
            if file_format == 'json':
                self.results_df.to_json(output_path, orient='records', lines=True)
            elif file_format == 'csv':
                self.results_df.to_csv(output_path, index=False)
            else:
                raise ValueError("Unsupported file format. Use 'json' or 'csv'.")
        else:
            raise ValueError("Results are not available. Please run find_top_similar first.")

    def get_results(self):
        return self.results_df