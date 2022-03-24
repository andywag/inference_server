import {Button, Grid} from '@mui/material'
import React, { useState, useEffect } from 'react';

import {Table,TableBody,TableCell,TableContainer,TableHead,TableRow} from '@mui/material';
import {Paper,Checkbox}  from '@mui/material';
import moment from 'moment';

import ReactJson from 'react-json-view'

function ResultTable(props) {
  

  const query = () => {
    const requestOptions = {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    };
    var time = new Date().getTime()

    fetch("http://192.168.3.114:8101/results",requestOptions)
      .then(response => response.json())
      .then(r => {
          var resultMap = {}
          r.results.forEach((element,index) => {
            resultMap[element.uuid] = element
          });
          setResults(r.results)
          props.setResultMap(resultMap)
          //props.setData(resultMap[selected].results)
        })
  }


  const [results, setResults] = useState([]);



  useEffect(() => {
    const timer = setInterval(query, 5000);
    return () => clearInterval(timer);
  }, []);




    const isSelected = (name) => {
        return props.selected == name
    };
    
    const handleClick = (evt, name) => {
        props.setSelected(name);
        //props.setData(resultMap[name].results)
    }

    const getDate = (row) => {
      const date = new Date(row.status[row.status.length-1].time * 1000)
      return moment(date).format('MM-DD-h:mm');
    }

    const getHost = (row) => {
      const value = row
      if (value.hasOwnProperty('hostname')) {
        return value.hostname
      }
      return ""
    }

    const handleContextMenu = () => {
      console.log("Here")
    }



    return (
        <TableContainer component={Paper}>
            <Table size="small" aria-label="simple table onContextMenu={handleContextMenu}">
                <TableHead>
                    <TableRow>
                        <TableCell></TableCell>
                        <TableCell>Name</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Time</TableCell>
                        <TableCell>Host</TableCell>
                        <TableCell>Config</TableCell>
                    </TableRow>
                </TableHead>
            <TableBody>
          {results.map((row,index) => (
            <TableRow
              key={row.uuid}
              sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
            >
            <TableCell>
                <Checkbox
                    checked={isSelected(row.uuid)}
                    onClick={(event) => handleClick(event,row.uuid)}
                >
                </Checkbox>
                </TableCell>
              <TableCell align="left" scope="row">{row.description.name}</TableCell>
              <TableCell align="left">{row.status[row.status.length-1].status}</TableCell>
              <TableCell align="left">{getDate(row)}</TableCell>
              <TableCell align="left">{getHost(row)}</TableCell>
              <TableCell align="left"><ReactJson collapsed="True" src={row.description}/></TableCell>

            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
    );

}

export default ResultTable;
