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
            headers: { 'Content-Type': 'application/json' }
        };

        try
        {
            // Await the response -- send this straight to IEX
            return await axios.get(`${Config.LOCAL.apiUrl}/companies`, requestOptions);
        }
        catch(error)
        {
            // Log any error
            console.error(error);
        }
    }
}