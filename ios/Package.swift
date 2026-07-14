// swift-tools-version:5.7
// EdgeVDB Swift Package.
//
// The Swift sources wrap the EdgeVDB C core. The C core static library is
// distributed as an XCFramework built by build-xcframework.sh; build it
// first (or fetch a release artifact) so the binary target path exists:
//
//   ./build-xcframework.sh   # produces build/EdgeVDB.xcframework
//
import PackageDescription

let package = Package(
    name: "EdgeVDB",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
    ],
    products: [
        .library(name: "EdgeVDB", targets: ["EdgeVDB"]),
    ],
    targets: [
        .binaryTarget(
            name: "EdgeVDBCore",
            path: "build/EdgeVDB.xcframework"
        ),
        .target(
            name: "EdgeVDB",
            dependencies: ["EdgeVDBCore"],
            path: "Sources/EdgeVDB"
        ),
        .testTarget(
            name: "EdgeVDBTests",
            dependencies: ["EdgeVDB"],
            path: "Tests"
        ),
    ]
)
