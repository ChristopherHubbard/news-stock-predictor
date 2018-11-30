import axios, { AxiosResponse, AxiosError, AxiosRequestConfig } from 'axios';
import Config from '../config';

// Abstract since totally static class
export abstract class PredictionService
{
    // Static async method to get all the companies
    public static async predict(index: string): Promise<any>
    {
        // Options for the post to add the user -- type?
        const requestOptions: any =
        {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            body: index
        };

        try
        {
            // Await the response -- send this to the prediction endpoint to run the model for this index
            return await axios.get(`${Config.apiUrl}/`, requestOptions);
        }
        catch(error)
        {
            // Log any error
            console.error(error);
        }
    }
}