FROM node:18-alpine

# Set the working directory inside the client folder
WORKDIR /app

# Copy package.json and package-lock.json first to leverage Docker cache
COPY package*.json ./

# Install dependencies
RUN npm install --frozen-lockfile

# Copy the rest of the application files
COPY . .

# Build the React app
RUN npm run build

# Serve the app using a lightweight server
RUN npm install -g serve
CMD ["serve", "-s", "build", "-l", "3000"]

# Expose the port
EXPOSE 3000