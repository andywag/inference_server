
import {FormControl, FormControlLabel, FormLabel, RadioGroup, Radio, TextField, Select, MenuItem, InputLabel} from '@mui/material';
import {Button, Grid, Box} from '@mui/material'

function SimpleText(props) 
{
    return (
        <Grid item xs={props.grid}>
            <TextField label={props.label} size="small" value={props.value} InputLabelProps={{ shrink: true }} onChange={(e) => {props.set(e.target.value);}}/>
        </Grid>
    )
}

function SimpleSelect(props) 
{
    
    return (
        <Grid item xs={props.grid}>
            <TextField select label={props.label} value={props.value} size="small" onChange={(e) => {props.set(e.target.value);}}>
                {props.options.map(element => {
                    return <MenuItem size="small" value={element}>{element}</MenuItem>
                })}
            </TextField>
        </Grid>
    )
}

export {SimpleText, SimpleSelect};
