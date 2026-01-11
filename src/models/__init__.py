from .sklearn_wrapper import (
    BaseModel,
    SklearnModel,
    LogisticModel,
    SVMModel,
    LDAModel,
    RidgeModel
)

# from .pipeline import (
#     MLflowPipeline
# )

__all__ = [ 'BaseModel'
            'SklearnModel',
            'LogisticModel',
            'SVMModel',
            'LDAModel',
            'RidgeModel'
            # 'MLflowPipeline'
            ]

BaseModelType: type[BaseModel] = BaseModel
SklearnTrainerType: type[SklearnModel] = SklearnModel
LogisticModelType: type[LogisticModel] = LogisticModel
SVMModelType: type[SVMModel] = SVMModel
LDAModelType: type[LDAModel] = LDAModel
RidgeModelType: type[RidgeModel] = RidgeModel


# MLflowPipelineType: type[MLflowPipeline] = MLflowPipeline