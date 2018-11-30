import { indexConstants } from '../constants';
import { IAction } from '../models';
import { IndexService } from '../services';
import { alertActions } from './alert.actions';
import { Dispatch } from 'redux';
import { history } from '../services';
import { Company } from '../models/company.model';

// Interfaces for the models
interface IIndexActions
{
    getCompanyList: () => ((dispatch: Dispatch<any>) => void),
}

// Export the user actions
export const indexActions: IIndexActions =
{
    getCompanyList: getCompanyList
};

// Function to create register user
function getCompanyList(): (dispatch: Dispatch<any>) => void
{
    return async (dispatch: Dispatch<any>) => 
    {
        dispatch(<IAction> {
            type: indexConstants.GET_COMPANIES_REQUEST
        });

        // Check type for this response
        try
        {
            // Should be a post so no real response
            const companies: Array<Company> = await IndexService.getCompanyList();

            // Send success dispatches -- include the company list
            dispatch(<IAction> {
                type: indexConstants.GET_COMPANIES_SUCCESS,
                companies: companies
            });

            // Dispatch the sucess
            dispatch(alertActions.success('Companies retrieved'));
        }
        catch(error)
        {
            // Dispatch error actions
            dispatch(<IAction> {
                type: indexConstants.GET_COMPANIES_ERROR,
                error: error
            });
            dispatch(alertActions.error('Failed to retrieve companies'));
        }
    }
}