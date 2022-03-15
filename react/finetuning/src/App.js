import logo from './logo.svg';
import './App.css';
import 'react-tabs/style/react-tabs.css';

import Main from './Main.js'
import FineTuning from './FineForm.js'

import { BrowserRouter, Router, Route, Routes, Link } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Menu from '@mui/material/Menu';
import Button from '@mui/material/Button';
import { useNavigate } from "react-router-dom";
import Box from '@mui/material/Box';
import FineTop from './FineTop'

import {
  createStyles,
  makeStyles,
  ThemeProvider,
  createTheme
} from "@mui/material/styles"
import { MenuItem } from '@mui/material';




function App() {


  return (
    <ThemeProvider theme={createTheme()}>
      
      <BrowserRouter>
        <AppBar position="static">
          <Container maxWidth="xl">
            <Toolbar disableGutters>
              
              <Typography>Graphcore Fine Tuning API</Typography>
              <Box sx={{ml:5}}><Typography><Link to="/">HOME</Link></Typography></Box>
              <Box sx={{ml:5}}><Typography><Link to="/finetuning">Tuning</Link></Typography></Box>

            </Toolbar>

          </Container>
        </AppBar>
        <Box sx={{ml:5}} >
        <Routes>
          <Route path="/" element={<Main/>}/>
          <Route path="/finetuning" element={<FineTop/>}/> 
        </Routes>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
    
  );
}

export default App;
