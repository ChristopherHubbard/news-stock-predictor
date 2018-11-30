import * as React from 'react';
import { connect, DispatchProp } from 'react-redux';
import { indexActions } from '../actions';
import { Company } from '../models';

interface HomeProps
{
    companies: Array<Company>
}

class HomePage extends React.Component<HomeProps & DispatchProp<any>, {}>
{
    constructor(props: HomeProps & DispatchProp<any>)
    {
        super(props);
        
        // Get the dispatch function from the props
        const { dispatch } = this.props;

        // On initialization set dispatch the action to get the indicies and then send the dispatch action
        dispatch(indexActions.getCompanyList());
    }

    public render(): React.ReactNode
    {
        // Extract prop data
        const { companies } = this.props;

        // Render the props on the combobox
        return (
            <div>
                <select>
                    {companies.map((company, i) => <option key={i}> {company.symbol}: {company.name} </option>)}
                </select>
            </div>
        )
    }
}

function mapStateToProps(state: any): HomeProps
{
    // Extract the state from the action
    const { companies } = state.index;
    return {
        companies
    };
}

export default connect<HomeProps>(
    mapStateToProps
)(HomePage);