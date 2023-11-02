import torch


retrieval_model = None


def build_retrieval_model():
    from sentence_transformers import SentenceTransformer

    global retrieval_model
    if retrieval_model is None:
        retrieval_model = SentenceTransformer("all-MiniLM-L6-v2").cuda()
        retrieval_model.eval()
    return retrieval_model


@torch.no_grad()
def build_dpr_index(dataset, key="question", batch_size=1024):
    is_single = False
    if isinstance(dataset, dict):
        dataset = [dataset]
        is_single = True

    assert isinstance(dataset, list), f"dataset is not a list, but {type(dataset)}"

    dpr_model = build_retrieval_model()
    embeddings = []
    query_text = [_[key] for _ in dataset]
    assert isinstance(query_text[0], str), f"key {key} is not a string, but {type(query_text[0])}"

    for i in range(0, len(dataset), batch_size):
        batch_size = min(batch_size, len(dataset) - i)
        embedding_i = dpr_model.encode(query_text[i : i + batch_size], convert_to_tensor=True)
        embeddings.append(embedding_i)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings[0] if is_single else embeddings


@torch.no_grad()
def dpr_query(target_question, source_embeddings, embedding_ratios, num_samples=5):
    from sentence_transformers.util import cos_sim

    dpr_model = build_retrieval_model()
    scores = 0
    score_dict = {}
    for key, embeddings in source_embeddings.items():
        query_embedding = dpr_model.encode([target_question[key]], convert_to_tensor=True)
        score_key = cos_sim(query_embedding, embeddings)
        score_dict[key] = score_key
        scores = scores + score_key * embedding_ratios[key]
    indices = scores[0].argsort(descending=True)[:num_samples]
    indices = [_.item() for _ in indices]

    score_dict["final"] = scores
    ret_scores = {key + "_score": [score_dict[key][0, i].item() for i in indices] for key in score_dict}
    return indices, ret_scores
