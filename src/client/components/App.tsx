import * as React from 'react';
import { Router, Route, HashRouter } from 'react-router-dom';
import HomePage from './HomePage';
import { PrivateRoute } from './PrivateRoute';
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
                        <Router history={history}>
                            <div>
                                <PrivateRoute exact path="/" component={HomePage}/>
                            </div>
                        </Router>
                    </div>
                </div>
            </div>
        );
    }
}