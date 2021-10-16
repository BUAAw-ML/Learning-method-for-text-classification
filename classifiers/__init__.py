from classifiers.GCNBert import GCNBert
from classifiers.LABert import LABert
from classifiers.MLPBert import MLPBert

# built_in_clf = {
#     "GCNBert": GCNBert,
#     "MABert": MABert,
#     "MLPBert": MLPBert
# }

def get_customized_clf(model_config):

    if model_config["clf_name"] == "GCNBert":
        pass

    elif model_config["clf_name"] == "LABert":
        pass

    elif model_config["clf_name"] == "MLPBert":
        model = MLPBert(model_config['output_dim'], model_config['hidden_dim'], model_config['hidden_layer_num'], 
                model_config['bert_trainable'])
    
    return model


# __all__ = [
#     "BertClassifier",
#     "EditDistanceMetric",
#     "GloVeSemanticSimilarityMetric",
#     "GPT2GrammarQualityMetric",
#     "MetricBase",
#     "USESemanticSimilarityMetric",
#     "SBERTSemanticSimilarityMetric",
#     "MetricBundle"]
