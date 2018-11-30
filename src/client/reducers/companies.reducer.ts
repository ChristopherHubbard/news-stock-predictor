import { indexConstants } from '../constants';
import { CompanyState, IAction } from '../models';

export function companies(state: CompanyState = { companies: [] }, action: IAction): CompanyState
{
    // Go through possible states for authentication
    switch (action.type)
    {
        case indexConstants.GET_COMPANIES_SUCCESS:
            return <CompanyState> {
                companies: action.companies
            };
        case indexConstants.GET_COMPANIES_ERROR:
            return <CompanyState> {
                companies: []
            };
        default:
            return state;
    }
}