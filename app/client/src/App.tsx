import React, { useState } from "react";
import "./App.css";
import Navbar from "./components/Navbar/Navbar";
import Home from "./pages/Home/Home";

const App: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState<boolean>(
    localStorage.getItem("theme") === "dark"
  );

  const toggleTheme = () => {
    setIsDarkMode((prevMode) => {
      const newMode = !prevMode;
      localStorage.setItem("theme", newMode ? "dark" : "light");
      return newMode;
    });
  };

  return (
    <div className={isDarkMode ? "dark-theme" : "light-theme"}>
      <Navbar isDarkMode={isDarkMode} toggleTheme={toggleTheme} />
      <div className="content">
        <Home />
      </div>
    </div>
  );
};

export default App;
