import React from 'react';

const Footer = () => {
  return (
    <footer style={styles.footer}>
      
      <div style={styles.textContainer}>
        <p style={styles.copyright}>© 2024 Universal-Box</p>
      </div>
    </footer>
  );
};

const styles = {
  footer: {
    backgroundColor: '#f8f9fa',
    padding: '20px',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    flexDirection: 'column',
    borderTop: '1px solid #e7e7e7',
    position: 'fixed',
    bottom: 0,
    width: '100%',
  },
  logoContainer: {
    marginBottom: '10px',
  },
  logo: {
    width: '50px',
    height: '50px',
  },
  textContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  text: {
    fontSize: '16px',
    color: '#555',
  },
  copyright: {
    fontSize: '16px',
    color: '#555',
    alignItems: 'left',
  },
};

export default Footer;