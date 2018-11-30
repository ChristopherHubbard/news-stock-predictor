import { Company } from './company.model';
import { Prediction } from './prediction.model'

export interface CompanyState
{
    companies?: Array<Company>
}

export interface PredictionState
{
    predicting?: boolean,
    prediction?: Prediction // Already in too many places
}