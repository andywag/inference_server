import {Button, Grid, Box} from '@mui/material'
import { useState, useEffect } from 'react';


import FineForm from './FineForm'
import ResultTable from './ResultTableGrid'
import ResultChart from './ResultChart'
import FineTable from './FineTable';
import ReactJson from 'react-json-view'

function FineTop(props) {
    
    const [resultMap, setResultMap] = useState({});
    const [selected, setSelected] = useState("");

    const [description, setDescription] = useState({});
    const [results, setResults] = useState({});

    var data = [];

    

    return (
        <Grid container spacing={2} >
            <Grid item xs={12}>
                <FineForm/>
            </Grid>
            <Grid item xs={12}>
                <FineTable setDescription= {setDescription} setResults={setResults}/>
            </Grid>
            <Grid item xs={3}>
              <p><b>Configuration</b></p>
               <ReactJson collapsed="True" src={description}/>
            </Grid>

            <Grid item xs={3}>
                <p><b>Results</b></p>
                <ReactJson collapsed="True" src={{results}}/>
            </Grid>
            <Grid item xs={12}>
                <ResultChart data = {results} description={description}/>
            </Grid>
       </Grid>
    )

}

export default FineTop;