"""
IR Project - Person 3 (J Sai Varun)
Learning-to-Rank Module (A+ Extension)
Implements supervised ranking using XGBoost
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.IR_core.config import Config
from core.IR_core.preprocessing import preprocess_text


def preprocess_query(text: str, config: Config) -> List[str]:
    """Use the same preprocessing settings that were used to build the index."""
    return preprocess_text(
        text=text,
        lowercase=config.lowercase,
        remove_punct=config.remove_punct,
        remove_digits=config.remove_digits,
        remove_stopwords=config.remove_stopwords,
        use_stemming=config.use_stemming and not config.use_lemmatize,
        use_lemmatize=config.use_lemmatize,
        min_token_len=config.min_token_len,
        max_token_len=config.max_token_len,
    )


class FeatureExtractor:
    """Extract features for query-document pairs"""
    
    def __init__(self, inverted_index, doc_stats, term_stats, doc_store_path):
        self.inverted_index = inverted_index
        self.term_stats = term_stats
        self.doc_len = {int(k): v for k, v in doc_stats['doc_len'].items()}
        self.avgdl = doc_stats['avgdl']
        
        # Load document metadata
        self.doc_metadata = {}
        with open(doc_store_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                self.doc_metadata[str(doc['doc_id'])] = doc
    
    def extract_features(self, query_tokens: List[str], doc_id: str,
                        bm25_score: float, tfidf_score: float) -> Dict[str, float]:
        """
        Extract features for a (query, document) pair
        
        Features:
        1. BM25 score
        2. TF-IDF score  
        3. Document length (normalized)
        4. Query length
        5. Query-document term overlap (Jaccard)
        6. Title match score
        7. IDF sum of query terms
        8. Average TF of query terms in doc
        9. Document length ratio (doc_len / avgdl)
        10. BM25 * doc_length_norm
        """
        doc_id_int = int(doc_id)
        doc_id_str = str(doc_id)

        features = {}
        
        # Feature 1 & 2: Ranking scores from base models
        features['bm25_score'] = bm25_score
        features['tfidf_score'] = tfidf_score
        
        # Feature 3: Document length (normalized)
        doc_len = self.doc_len.get(doc_id_int, 0)
        features['doc_length_norm'] = doc_len / self.avgdl if self.avgdl > 0 else 0
        
        # Feature 4: Query length
        features['query_length'] = len(query_tokens)
        
        # Feature 5: Query-document term overlap (Jaccard similarity)
        doc_terms = set()
        query_unique = set(query_tokens)
        tf_values = []
        for term in query_tokens:
            postings = self.inverted_index.get(term)
            if postings and doc_id_int in postings:
                doc_terms.add(term)
                tf_values.append(postings[doc_id_int]["tf"])
        
        intersection = len(doc_terms)
        union = len(query_unique | doc_terms)
        features['term_overlap_jaccard'] = intersection / union if union > 0 else 0
        
        # Feature 6: Title match score
        if doc_id_str in self.doc_metadata:
            title = self.doc_metadata[doc_id_str].get('title', '').lower()
            title_tokens = set(title.split())
            title_match = len(query_unique & title_tokens) / len(query_unique) if query_unique else 0
            features['title_match_score'] = title_match
        else:
            features['title_match_score'] = 0
        
        # Feature 7: IDF sum of query terms
        idf_sum = sum(self.term_stats['idf'].get(term, 0) for term in query_tokens)
        features['query_idf_sum'] = idf_sum
        
        # Feature 8: Average TF of query terms in document
        features['avg_term_frequency'] = np.mean(tf_values) if tf_values else 0
        
        # Feature 9: Document length ratio
        features['doc_length_ratio'] = doc_len / self.avgdl if self.avgdl > 0 else 1.0
        
        # Feature 10: Combined score (BM25 weighted by doc length)
        features['bm25_length_product'] = bm25_score * features['doc_length_norm']
        
        return features


class LTRTrainer:
    """Learning-to-Rank trainer using XGBoost"""
    
    def __init__(self, feature_extractor: FeatureExtractor, query_config: Config):
        self.feature_extractor = feature_extractor
        self.query_config = query_config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def prepare_training_data(self, 
                            queries_file: str,
                            tfidf_results: Dict[str, List[str]],
                            bm25_results: Dict[str, List[str]],
                            tfidf_ranker,
                            bm25_ranker) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from queries and relevance judgments
        
        Args:
            queries_file: Path to queries JSON with relevance judgments
            tfidf_results: Results from TF-IDF ranker
            bm25_results: Results from BM25 ranker
            tfidf_ranker: TF-IDF ranker instance (to get scores)
            bm25_ranker: BM25 ranker instance (to get scores)
        
        Returns:
            X (features), y (relevance labels), groups (query grouping for ranking)
        """
        # Load queries
        with open(queries_file, 'r') as f:
            queries = json.load(f)
        
        X_data = []
        y_data = []
        groups = []
        
        for query in queries:
            query_id = query['query_id']
            query_text = query['query_text']
            relevance_judgments = query['relevance_judgments']
            
            # Preprocess query
            query_tokens = preprocess_query(query_text, self.query_config)
            if not query_tokens:
                continue

            tfidf_score_map = tfidf_ranker.rank(query_tokens)
            bm25_score_map = bm25_ranker.rank(query_tokens)
            
            # Get all candidate documents (union of top results from both models)
            candidate_docs = set()
            if query_id in tfidf_results:
                candidate_docs.update(int(doc) for doc in tfidf_results[query_id][:50])
            if query_id in bm25_results:
                candidate_docs.update(int(doc) for doc in bm25_results[query_id][:50])
            
            query_group_size = 0
            
            # Extract features for each candidate document
            for doc_id in candidate_docs:
                # Get scores from both rankers
                doc_id_int = int(doc_id)
                doc_id_str = str(doc_id_int)
                tfidf_score = tfidf_score_map.get(doc_id_int, 0.0)
                bm25_score = bm25_score_map.get(doc_id_int, 0.0)
                
                # Extract features
                features = self.feature_extractor.extract_features(
                    query_tokens, doc_id_int, bm25_score, tfidf_score
                )
                
                # Get relevance label
                relevance = relevance_judgments.get(doc_id_str, relevance_judgments.get(doc_id_int, 0))
                
                X_data.append(list(features.values()))
                y_data.append(relevance)
                query_group_size += 1
                
                if self.feature_names is None:
                    self.feature_names = list(features.keys())
            
            groups.append(query_group_size)
        
        X = np.array(X_data)
        y = np.array(y_data)
        groups = np.array(groups)
        
        return X, y, groups
    
    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              test_size: float = 0.2, random_state: int = 42):
        """
        Train the XGBoost ranking model
        
        Args:
            X: Feature matrix
            y: Relevance labels
            groups: Query group sizes
            test_size: Proportion for test set
            random_state: Random seed
        """
        # Split data preserving query groups
        # For simplicity, we'll do a basic split here
        # In production, you'd want to ensure complete queries stay together
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)
        
        # XGBoost parameters for ranking
        params = {
            'objective': 'rank:ndcg',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'ndcg@10',
            'seed': random_state
        }
        
        # Train model
        print("Training XGBoost LTR model...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        print("\nTraining complete!")
        
        # Feature importance
        if self.feature_names:
            importance = self.model.get_score(importance_type='gain')
            print("\nFeature Importance:")
            for feat_name, imp_score in sorted(importance.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True):
                feat_idx = int(feat_name.replace('f', ''))
                if feat_idx < len(self.feature_names):
                    print(f"  {self.feature_names[feat_idx]:<25}: {imp_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict relevance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        dmatrix = xgb.DMatrix(X_scaled)
        return self.model.predict(dmatrix)
    
    def save_model(self, model_path: str, scaler_path: str):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.save_model(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        feature_names_path = Path(model_path).parent / 'feature_names.json'
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path: str, scaler_path: str):
        """Load trained model and scaler"""
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature names
        feature_names_path = Path(model_path).parent / 'feature_names.json'
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
        
        print(f"Model loaded from {model_path}")


class LTRRanker:
    """Ranker that uses trained LTR model"""
    
    def __init__(self, ltr_trainer: LTRTrainer, 
                 tfidf_ranker, bm25_ranker,
                 feature_extractor: FeatureExtractor):
        self.ltr_trainer = ltr_trainer
        self.tfidf_ranker = tfidf_ranker
        self.bm25_ranker = bm25_ranker
        self.feature_extractor = feature_extractor
    
    def rank(self, query_text: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Rank documents using LTR model
        
        Args:
            query_text: Query string
            top_k: Number of results to return
        
        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        query_tokens = preprocess_query(query_text, self.ltr_trainer.query_config)
        if not query_tokens:
            return []
        
        # Get candidate documents from base rankers
        tfidf_scores = self.tfidf_ranker.rank(query_tokens)
        bm25_scores = self.bm25_ranker.rank(query_tokens)
        
        def top_docs(score_map):
            return sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        candidates = {}
        for doc_id, score in top_docs(tfidf_scores):
            candidates[doc_id] = {'tfidf': score, 'bm25': 0.0}
        for doc_id, score in top_docs(bm25_scores):
            if doc_id in candidates:
                candidates[doc_id]['bm25'] = score
            else:
                candidates[doc_id] = {'tfidf': 0.0, 'bm25': score}
        
        if not candidates:
            return []
        
        # Extract features for all candidates
        X_data = []
        doc_ids = []
        
        for doc_id, scores in candidates.items():
            features = self.feature_extractor.extract_features(
                query_tokens, doc_id, scores['bm25'], scores['tfidf']
            )
            X_data.append(list(features.values()))
            doc_ids.append(doc_id)
        
        if not X_data:
            return []
        
        # Predict scores
        X = np.array(X_data)
        scores = self.ltr_trainer.predict(X)
        
        # Sort by predicted score
        results = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


if __name__ == "__main__":
    print("LTR module loaded successfully!")
    print("\nKey classes:")
    print("  - FeatureExtractor: Extract features for query-doc pairs")
    print("  - LTRTrainer: Train XGBoost ranking model")
    print("  - LTRRanker: Rank documents using trained LTR model")