import { Dispatch } from 'redux';
import { PredictionService } from '../services';
import { predictionConstants } from '../constants';
import { alertActions } from './alert.actions';

interface IPredictionActions
{
    predict: (index: string) => (dispatch: Dispatch<any>) => void,
}

export const coinActions: IPredictionActions =
{
    predict: predict
};

function predict(index: string): (dispatch: Dispatch<any>) => void
{
    return async (dispatch) =>
    {
        // Set that the assets are being retrieved
        dispatch({
            type: predictionConstants.GET_PREDICTION_REQUEST,
            index: index
        });

        try
        {
            const prediction = await PredictionService.predict(index);

            dispatch({
                type: predictionConstants.GET_PREDICTION_SUCCESS,
                prediction: prediction
            });

            dispatch(alertActions.success("Get Prediction Sucess"));
        }
        catch(error)
        {
            dispatch({
                type: predictionConstants.GET_PREDICTION_ERROR,
                error: error
            });

            dispatch(alertActions.error("Get Prediction Error"));
        }
    }
}