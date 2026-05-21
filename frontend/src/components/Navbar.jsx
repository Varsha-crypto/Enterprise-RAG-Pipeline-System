import React from 'react';
import './Navbar.css';

const Navbar = ({ onLogoClick }) => {
    return (
        <nav className="navbar">
            <div className="navbar-container">
                <div className="navbar-left">
                    <div className="navbar-logo" onClick={onLogoClick} style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                        {/* Left Logo - JPG */}
                        <img src="/src/assets/logos/logo-left.jpg" alt="Left Logo" className="nav-logo-img left-logo" onError={(e) => e.target.style.display='none'} />
                        <div className="logo-placeholder" style={{ display: 'flex', flexDirection: 'row', alignItems: 'baseline' }}>
                            <span className="logo-text-shakti">Shakti</span>
                            <span className="logo-text-db">DB</span>
                            <span className="logo-tm">™</span>
                        </div>
                    </div>
                </div>

                <div className="navbar-right">
                    {/* Right Logo */}
                    <img src="/src/assets/logos/logo-right.png" alt="Right Logo" className="nav-logo-img right-logo" onError={(e) => e.target.style.display='none'} />
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
