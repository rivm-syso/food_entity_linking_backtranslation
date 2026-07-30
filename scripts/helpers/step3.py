import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def get_all_positions_and_similarities(orig_names, processor):
    # Make a mapping from name to index
    # E.g. {'Yoghurt': 1500} to find at which positions (indices) the orig_name is
    if isinstance(processor.df1, pd.Series):  # For BoW and tf-idf
        name_to_index = {name: idx for idx, name in enumerate(processor.df1)}
    else:
        name_to_index = {name: idx for idx, name in enumerate(processor.df1['name'])}
    orig_indices = np.array([name_to_index[name] for name in orig_names])

    # Transpose similarity matrix to get FoodOn x (nevo/fdc/kap)
    similarity_matrix = processor.similarity_matrix.T

    # Sort indices per row on descending similarity, i.e. per row / FoodOn item, the first element is the index of the most similar nevo/fdc/kap item
    sorted_indices = np.flip(np.argsort(similarity_matrix, axis=1), axis=1)

    # all_positions, i.e. for each FoodOn item, a list of len(orig_indices) stating per food item in the (nevo/fdc/kap) subset the position p,
    # where p states that the food item from the subset is the p'th most similar food item to the FoodOn item.
    # shape FoodOn x len(orig_indices)
    all_positions = (sorted_indices[:, :, None] == orig_indices).argmax(axis=1)
    
    # all_similarities, i.e. the similarity score per food item in the (nevo/fdc/kap) subset for each FoodOn item
    # shape FoodOn x len(orig_indices)
    all_similarities = similarity_matrix[:, orig_indices]

    return all_positions, all_similarities

def sort_on_positions_and_similarities(all_positions, all_similarities):
    # all_positions, all_similarities: shape (n_rows, n_orig_names)
    n_rows, n_orig_names = all_positions.shape
    all_sorted_lists = []
    # Determine per food item in the subset the new ranking
    for i in range(n_orig_names):
        df_possim = pd.DataFrame({
            'id': np.arange(n_rows),
            'position': all_positions[:, i], # The positions per FoodOn item
            'similarity': all_similarities[:, i] # The similarities per FoodOn item
        })
        # Sort on position (index) first (ascending), then similarity (descending), e.g. lower index is better, higher similarity is better.
        df_possim_sorted = df_possim.sort_values(by=['position', 'similarity'], ascending=[True, False])
        all_sorted_lists.append(df_possim_sorted)
    return all_sorted_lists

def get_all_position_matches_and_sorted_lists(df_merged_data, processor, column_name='nevo_name'):
    orig_names = df_merged_data[column_name].tolist()
    # Vectorized positions and similarities
    all_positions, all_similarities = get_all_positions_and_similarities(orig_names, processor)
    all_sorted_lists = sort_on_positions_and_similarities(all_positions, all_similarities)
    # Add candidates
    if isinstance(processor.df1, pd.Series):  # For BoW and tf-idf
        candidate_names = np.array(processor.df2)
    else:
        candidate_names = np.array(processor.df2['name'])
    for df in all_sorted_lists:
        df['candidates'] = candidate_names[df['id'].values]
    # Determine position of best match
    all_position_matches = []
    for i, orig_name in enumerate(orig_names):
        sorted_list = all_sorted_lists[i]
        df_merged_filter = df_merged_data[df_merged_data[column_name] == orig_name].reset_index()
        best_match = df_merged_filter['name'][0] if pd.notna(df_merged_filter['name'][0]) else df_merged_filter['best_match'][0]
        candidates = list(sorted_list['candidates'])
        position_match = candidates.index(best_match) if best_match in candidates else np.nan
        all_position_matches.append(position_match)
    return all_sorted_lists, all_position_matches

def top_k_from_positions(position_matches, k):
    return sum(np.array(position_matches) < k) # < because position_match is an index

def top_k_results(merged_data, position_column, labels = ['broader', 'exact', 'close match', 'related', 'narrow'], only_perc=False):
    merged_data = merged_data[merged_data['final_label'].isin(labels)]
    n_matches = merged_data.shape[0]
    position_matches = merged_data[position_column]
    top_1, top_3, top_5, top_15 = [top_k_from_positions(position_matches, k) for k in [1,3,5,15]]
    if n_matches == 0:
        if only_perc:
            return ['0.0%', '0.0%', '0.0%', '0.0%']
        else:
            return ['0/0 (0.0%)', '0/0 (0.0%)', '0/0 (0.0%)', '0/0 (0.0%)']
    if only_perc:
        return [str(round(top_1/n_matches*100, 1)) + "%",
                str(round(top_3/n_matches*100, 1)) + "%",
                str(round(top_5/n_matches*100, 1)) + "%",
                str(round(top_15/n_matches*100, 1)) + "%"]
    else:
        return [str(top_1) + "/" + str(n_matches) + " (" + str(round(top_1/n_matches*100, 1)) + "%)",
                str(top_3) + "/" + str(n_matches) + " (" + str(round(top_3/n_matches*100, 1)) + "%)",
                str(top_5) + "/" + str(n_matches) + " (" + str(round(top_5/n_matches*100, 1)) + "%)",
                str(top_15) + "/" + str(n_matches) + " (" + str(round(top_15/n_matches*100, 1)) + "%)"]


def get_orig_candidates(orig_name, processor):
    if isinstance(processor.df1, pd.Series):  # For BoW and tf-idf
        # Get original names
        names = processor.df1.values
        try:
            # Get index of orig_name
            orig_index = np.where(names == orig_name)[0][0]
        except IndexError:
            raise ValueError(f"{orig_name} not found in df1")
        # Use index to sort the candidates of orig_name from most similar to least similar
        orig_ranking = np.flip(np.argsort(processor.similarity_matrix[orig_index]))
        orig_candidates = processor.df2.iloc[orig_ranking].values
    else:
        # Get original names
        names = processor.df1['name'].values
        try:
            # Get index of orig_name
            orig_index = np.where(names == orig_name)[0][0]
        except IndexError:
            raise ValueError(f"{orig_name} not found in df1['name']")
        # Use index to sort the candidates of orig_name from most similar to least similar
        orig_ranking = np.flip(np.argsort(processor.similarity_matrix[orig_index]))
        orig_candidates = processor.df2.iloc[orig_ranking]['name'].values
    return orig_candidates

def combine_results(orig_candidates, reranked_candidates, similarities):
    # Make a dict for fast lookup
    reranked_pos_dict = {candidate: pos for pos, candidate in enumerate(reranked_candidates)}
    orig_positions = np.arange(len(orig_candidates))
    reranked_positions = np.array([reranked_pos_dict.get(candidate, np.nan) for candidate in orig_candidates])
    hybrid_positions = (orig_positions + reranked_positions) / 2

    df_candidates = pd.DataFrame({
        'original': orig_candidates,
        'reranked': reranked_candidates,
        'reranked_similarities': similarities,
        'orig_positions': orig_positions,
        'reranked_positions': reranked_positions,
        'hybrid_positions': hybrid_positions
    })

    return df_candidates.sort_values(by='hybrid_positions').reset_index(drop=True)

def get_hybrid_position_match(df_merged_data, processor, all_sorted_lists, column_name='nevo_name'):
    all_orig_candidates = []
    for orig_name in df_merged_data[column_name]:
        orig_candidates = get_orig_candidates(orig_name, processor)
        all_orig_candidates.append(orig_candidates)

    all_hybrid_results = []
    for i, orig_candidates in enumerate(all_orig_candidates):
        reranked_candidates = all_sorted_lists[i]['candidates']
        similarities = all_sorted_lists[i]['similarity']
        df_hybrid_result = combine_results(orig_candidates, reranked_candidates, similarities)
        all_hybrid_results.append(df_hybrid_result)

    best_match_plus_proposed = df_merged_data['name'].where(
        df_merged_data['name'].notna(), df_merged_data['best_match']
    ).values

    hybrid_position_match = []
    for i, best_match in enumerate(best_match_plus_proposed):
        orig = all_hybrid_results[i]['original'].values
        try:
            pos = np.where(orig == best_match)[0][0]
        except IndexError:
            pos = np.nan
        hybrid_position_match.append(pos)

    return best_match_plus_proposed, hybrid_position_match