import axios, { AxiosResponse, AxiosError, AxiosRequestConfig } from 'axios';
import Config from '../config';

// Abstract since totally static class
export abstract class IndexService
{
    // Static async method to get all the companies
    public static async getCompanyList(): Promise<any>
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
                env: 'DEV'
            }
        };

        try
        {
            // Await the response -- Gets from Dynamo
            const response: AxiosResponse = await axios.get(`${Config.DEV.apiUrl}/companies`, requestOptions);

            // Transform the response for the redux actions
            return response.data.companies;
        }
        catch(error)
        {
            // Log any error
            console.error(error);
        }
    }
}