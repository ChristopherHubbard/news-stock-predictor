import { alertConstants } from '../constants';
import { IAction } from '../models';

// Define interface model
interface IAlertActions
{
    success: (message: string) => IAction
    error: (message: string) => IAction
    clear: (message: string) => IAction
}

// Define exported object
export const alertActions: IAlertActions =
{
    success: success,
    error: error,
    clear: clear
};

// Define redux action functions
function success(message: string): IAction
{
    return {
        type: alertConstants.SUCCESS,
        message: message
    };
}

function error(message: string): IAction
{
    return {
        type: alertConstants.ERROR,
        message: message
    };
}

function clear(message: string): IAction
{
    return {
        type: alertConstants.CLEAR,
        message: message
    }
}
