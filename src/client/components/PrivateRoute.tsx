import { Route, Redirect } from 'react-router-dom';
import * as React from 'react';

// Create route that either creates component in render or redirects to the login page
export const PrivateRoute = ({ component: Component, ...rest }: any) => (
    <Route {...rest} render={ (props) => (
        localStorage.getItem('user') ? <Component {...props}/> : <Redirect to='/login'/>
    )}/>
);