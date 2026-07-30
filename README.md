# Back and Forth Food Entity Linking via Reranking based on Back-translation

Code associated to manuscript 'Back and Forth Food Entity Linking via Reranking based on Back-translation'

## Description

Food entity linking becomes increasingly useful for answering cross-discipline questions with diverse and heterogeneous food datasets, but it remains a challenging task. An automatic approach where entities in a source dataset are linked to a target entities using embedding similarities is fast but suboptimal. Therefore, we extend this approach by reranking entities by mapping back target candidates to the source dataset (target ranking), based on the back-translation concept, assuming that a match is better when it can be linked back. We combine the original approach with our target ranking to create a hybrid ranking, to create a back-and-forth mapping approach. We evaluated our approaches using four vector representation techniques on a reusable expert-curated annotated golden standard from diverse datasets that we make publicly available. Overall, the best match of an entity is ranked higher using the hybrid ranking compared to the original and target ranking on the golden standard. However, results vary per vector representation technique and per dataset. This underlines the non-trivial task of food entity linking, to which our approach can be a worthwhile strategy.

## Getting Started

### Setting up a Python environment from requirements.txt
To ensure you have all the necessary dependencies installed, it’s recommended to use a virtual environment and the provided requirements.txt file.

1. Create and activate a virtual environment
On macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

On Windows (PowerShell or cmd):
```
python -m venv venv
venv\Scripts\activate
```

Once activated, your shell prompt should show (venv).

2. Install dependencies from requirements.txt
From the project root (where requirements.txt is located), run:

```
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages into the virtual environment.

### Replicating the analysis
The code is split in helper functions (in the _helpers_ folder) and four python notebooks. The four notebooks contain the four steps of the analysis and should be run in order, starting with __step1_ontology_mapping_full_ ranking.ipynb__. In __step4_tables_and_figures.ipynb__, all tables and figures from the manuscript are recreated. This can be run after running the previous three steps.

## License

This project is licensed under the EUPL License - see the LICENSE file for details