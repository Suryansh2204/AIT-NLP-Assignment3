import React from "react";

interface NavbarProps {
  isDarkMode: boolean;
  toggleTheme: () => void;
}

const Navbar: React.FC<NavbarProps> = ({ isDarkMode, toggleTheme }) => {
  return (
    <nav className={`navbar ${isDarkMode ? "dark-navbar" : "light-navbar"}`}>
      <div className="site-title">LostInTranslation</div>
      <div style={{ marginRight: "3rem" }}>
        {/* <ThemeToggle isDarkMode={isDarkMode} toggleTheme={toggleTheme} /> */}
      </div>
    </nav>
  );
};

export default Navbar;
