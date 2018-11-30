import { predictionConstants } from '../constants';
import { PredictionState, IAction } from '../models';

export function prediction(state: PredictionState = {}, action: IAction): PredictionState
{
    // Go through possible states for authentication
    switch (action.type)
    {
        case predictionConstants.GET_PREDICTION_REQUEST:
            return <PredictionState> {
                predicting: true
            };
        case predictionConstants.GET_PREDICTION_SUCCESS:
            return <PredictionState> {
                predicting: false,
                prediction: action.prediction
            };
        case predictionConstants.GET_PREDICTION_ERROR:
            return <PredictionState> {}
        default:
            return state;
    }
}