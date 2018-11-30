import { Company } from "./company.model";
import { Prediction } from "./prediction.model";

// This IAction interface might want to be moved over to some interfaces folder
export interface IAction
{
    type: string,
    message?: string,
    error?: string,
    companies?: Array<Company>,
    prediction?: Prediction // What type? Binary classification
}