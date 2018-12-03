import * as React from 'react';
import HomePage from './HomePage';
import { history } from '../services';

export class App extends React.Component<{}, {}>
{
    constructor(props: any)
    {
        super(props);
    }

    public render(): React.ReactNode
    {
        return (
            <div className="jumbotron">
                <div className="container">
                    <div className="col-sm-8 col-sm-offset-2">
                        <div>
                            <HomePage/>
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}