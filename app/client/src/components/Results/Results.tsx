import React from "react";

interface ResultsProps {
  words: string[];
}

const Results: React.FC<ResultsProps> = ({ words }) => {
  return (
    <ul className="search-results">
      {words.length === 0 ? (
        <p>No results found</p>
      ) : (
        words.map((word, index) => (
          <li key={index} className="result-item">
            {word}
          </li>
        ))
      )}
    </ul>
  );
};

export default Results;
