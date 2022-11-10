import bentoml
from pydantic import BaseModel
import scipy.sparse as sp
from bentoml.io import JSON


class TwitterDisasterApp(BaseModel):
    location: str
    keyword: str
    text: str


model_ref = bentoml.sklearn.get("twitter-disasters-model:latest")

transformers = model_ref.custom_objects

model_runner = model_ref.to_runner()

svc = bentoml.Service("twitter_disasters_classifier", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=TwitterDisasterApp), output=JSON())
def classify(twitter_object):
    application_data = twitter_object.dict()

    c_vectorizer = transformers['c_vect']
    d_vectorizer = transformers['d_vect']
    p_features = transformers['p_feat']
    s_k_best = transformers['s_k_best']
    svd = transformers['svd']

    kw_loc_dict = {key: application_data[key] for key in ['location', 'keyword']}
    kw_loc_vectors = d_vectorizer.transform(kw_loc_dict)
    text_vectors = c_vectorizer.transform([application_data['text']])
    text_kw_loc_vectors = sp.hstack((text_vectors, kw_loc_vectors))
    text_kw_loc_poly_vectors = p_features.transform(text_kw_loc_vectors)
    best_vectors = s_k_best.transform(text_kw_loc_poly_vectors)

    vectors = svd.transform(best_vectors)

    prediction = model_runner.predict.run(vectors)
    return {
        'disaster': prediction == 1
    }