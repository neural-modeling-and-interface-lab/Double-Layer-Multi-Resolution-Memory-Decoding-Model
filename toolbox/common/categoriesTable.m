function category = categoriesTable(i)
%% Find the category name of it's number/index
switch (i)
    case 1
        category = 'Outline';
    case 2
        category = 'Silhouette';
    case 3
        category = 'Cartoon';
    case 4
        category = 'Abstract';
    case 5
        category = 'Natual';
    case 6
        category = 'Black&White';
    case 7
        category = 'Color';
    case 8
        category = 'Red';
    case 9
        category = 'Orange';
    case 10
        category = 'Yellow';
    case 11
        category = 'Green';
    case 12
        category = 'Blue';
    case 13
        category = 'Purple';
        %     case 14 % 2017-6-28 Delete Rainbow from the table
        %         category = 'Rainbow';
    case 14
        category = 'Activity';
    case 15
        category = 'Animal';
    case 16
        category = 'Artifact';
    case 17
        category = 'Building';
    case 18
        category = 'Face';
    case 19
        category = 'Food';
    case 20
        category = 'Geometric';
    case 21
        category = 'Human';
    case 22
        category = 'Landscape';
    case 23
        category = 'Plant';
    case 24
        category = 'Product Logo';
    case 25
        category = 'Symbol';
    case 26
        category = 'Tool';
    case 27
        category = 'Vehicle';
    case 28
        category = 'AnimalPrimary';
    case 29
        category = 'BuildingPrimary';
    case 30
        category = 'PlantPrimary';
    case 31
        category = 'ToolPrimary';
    case 32
        category = 'VehiclePrimary';
    case 33
        category = 'CorrectOrError';
    otherwise
        category = 'ERROR!';
end