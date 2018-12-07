import * as React from 'react';
import { connect, DispatchProp } from 'react-redux';
import { indexActions, predictionActions } from '../actions';
import { Company } from '../models';
import PredictionIndicator from './PredictionIndicator';

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

        // Bind methods
        this.onIndexChange = this.onIndexChange.bind(this);
    }

    private onIndexChange(event: React.ChangeEvent<HTMLSelectElement>): void
    {
        // When the index changes, dispatch the prediction action
        const { dispatch } = this.props;
        const selectedIndex = event.target.value;

        dispatch(predictionActions.predict(selectedIndex));
    }

    public render(): React.ReactNode
    {
        // Extract prop data
        const { companies } = this.props;

        // Sort before rendering
        const sortedCompanies: Array<Company> = 
            companies.sort((company1: Company, company2: Company) => company1.symbol > company2.symbol ? 1 : -1);

        // Render the props on the combobox -- Make sure there is no issue with map on empty array
        return (
            <div>
                Welcome to Stock Prediction!
                <select onChange={this.onIndexChange}>
                    {sortedCompanies !== undefined && sortedCompanies.length !== 0 ?
                        sortedCompanies.map(
                            (company, i) => 
                                <option key={i} value={company.symbol}>
                                    {company.symbol}: {company.name}
                                </option>
                        ) : null}
                </select>
                <PredictionIndicator/>
            </div>
        )
    }
}

function mapStateToProps(state: any): HomeProps
{
    // Extract the state from the action
    const { companies } = state.companies;
    return {
        companies
    };
}

export default connect<HomeProps>(
    mapStateToProps
)(HomePage);