# Use the official Golang image as the base image
FROM golang:1.24.1

# Set the working directory inside the container
WORKDIR /app

# Copy the Go application source code into the container
COPY . .

# Download and install dependencies
RUN go mod download

# Install Air for hot reload
RUN go install github.com/air-verse/air@latest

# Expose the port the application will run on
EXPOSE 8080

# Command to run the application with hot reload
CMD ["air"]