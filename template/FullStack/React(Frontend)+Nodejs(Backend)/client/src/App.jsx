import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home' 
import Signin from './pages/Signin'
import Signup from './pages/Signup'
import Header from './components/Header'
import ContactUs from './pages/ContactUs'
import TermsOfService from './pages/TermsOfService'
import  PrivacyPolicy from './pages/PrivacyPolicy'
import './index.css';
import Footer from './components/Footer'

export default function App() {
  return (
    <BrowserRouter>
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/signin" element={<Signin />} />
        <Route path="/signup" element={<Signup />} />
        <Route path='/contactus' element={<ContactUs/>}></Route>
        <Route path='/terms' element={<TermsOfService/>}></Route>
        <Route path='/privacy' element={<PrivacyPolicy/>}></Route>
      </Routes>
      <Footer />
    </BrowserRouter>
  )
}
