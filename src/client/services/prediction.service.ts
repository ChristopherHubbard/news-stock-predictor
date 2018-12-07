import axios, { AxiosResponse, AxiosError, AxiosRequestConfig } from 'axios';
import Config from '../config';

// Abstract since totally static class
export abstract class PredictionService
{
    // Static async method to get all the companies
    public static async predict(symbol: string): Promise<any>
    {
        // Options for the post to add the user -- type?
        const requestOptions: any =
        {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'
            },
            params: {
                symbol: symbol
            }
        };

        try
        {
            // Await the response -- send this to the prediction endpoint to run the model for this index
            const response: AxiosResponse = await axios.get(`${Config.DEV.apiUrl}/prediction`, requestOptions);

            return response.data.prediction;
        }
        catch(error)
        {
            // Log any error
            console.error(error);
        }
    }
}