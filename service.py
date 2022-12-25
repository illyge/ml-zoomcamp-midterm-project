import bentoml
from pydantic import BaseModel
from bentoml.io import JSON
import pandas as pd

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

    prediction = model_runner.predict.run(pd.DataFrame(data=application_data, index=[0]))
    return {
        'disaster': prediction == 1
    }